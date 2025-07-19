"""
Agent Manager - Central coordination and task distribution system.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import heapq
from core.base_agent import BaseAgent, Task, TaskStatus
from core.communication import MessageBroker
from config import settings, TASK_PRIORITIES
import structlog

logger = structlog.get_logger()

class TaskQueue:
    """Priority queue for managing agent tasks."""
    
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def put(self, task: Task) -> None:
        """Add a task to the priority queue."""
        # Use negative priority for max heap behavior
        heapq.heappush(self._queue, (-task.priority, self._index, task))
        self._index += 1
    
    def get(self) -> Optional[Task]:
        """Get the highest priority task from the queue."""
        if self._queue:
            _, _, task = heapq.heappop(self._queue)
            return task
        return None
    
    def size(self) -> int:
        """Get the current queue size."""
        return len(self._queue)
    
    def peek(self) -> Optional[Task]:
        """Peek at the highest priority task without removing it."""
        if self._queue:
            return self._queue[0][2]
        return None

class AgentManager:
    """Central agent manager for coordinating all agents and tasks."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = TaskQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.message_broker = MessageBroker()
        self.running = False
        self._task_dispatch_interval = 1.0  # seconds
    
    async def start(self) -> None:
        """Start the agent manager."""
        logger.info("Starting Agent Manager")
        self.running = True
        
        # Start background tasks
        asyncio.create_task(self._task_dispatcher())
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._cleanup_completed_tasks())
        
        logger.info("Agent Manager started successfully")
    
    async def stop(self) -> None:
        """Stop the agent manager."""
        logger.info("Stopping Agent Manager")
        self.running = False
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.status = TaskStatus.CANCELLED
        
        logger.info("Agent Manager stopped")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the manager."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the manager."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def submit_task(self, task_type: str, data: Dict[str, Any], priority: str = "medium") -> str:
        """Submit a new task to the system."""
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            priority=TASK_PRIORITIES.get(priority, 2),
            data=data,
            created_at=datetime.now()
        )
        
        self.task_queue.put(task)
        logger.info(f"Task submitted: {task.id} ({task_type}, priority: {priority})")
        
        return task.id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check completed tasks
        for task in self.completed_tasks + self.failed_tasks:
            if task.id == task_id:
                return self._task_to_dict(task)
        
        # Check pending tasks in queue
        for _, _, task in self.task_queue._queue:
            if task.id == task_id:
                return self._task_to_dict(task)
        
        return None
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific agent."""
        if agent_id in self.agents:
            return self.agents[agent_id].get_status()
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and metrics."""
        agent_states = defaultdict(int)
        for agent in self.agents.values():
            agent_states[agent.state] += 1
        
        return {
            "total_agents": len(self.agents),
            "agent_states": dict(agent_states),
            "pending_tasks": self.task_queue.size(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "system_uptime": datetime.now().isoformat(),
            "agents": [agent.get_status() for agent in self.agents.values()]
        }
    
    async def _task_dispatcher(self) -> None:
        """Background task for dispatching tasks to available agents."""
        while self.running:
            try:
                await self._dispatch_next_task()
                await asyncio.sleep(self._task_dispatch_interval)
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _dispatch_next_task(self) -> None:
        """Dispatch the next available task to a suitable agent."""
        task = self.task_queue.get()
        if not task:
            return
        
        # Find available agents that can handle this task
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == "idle" and agent.can_handle_task(task)
        ]
        
        if not available_agents:
            # No available agents, put task back in queue
            self.task_queue.put(task)
            return
        
        # Select the agent with the best performance (lowest failure rate)
        selected_agent = min(
            available_agents,
            key=lambda a: a.tasks_failed / max(a.tasks_completed + a.tasks_failed, 1)
        )
        
        # Assign task to agent
        self.active_tasks[task.id] = task
        logger.info(f"Dispatching task {task.id} to agent {selected_agent.agent_id}")
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task_with_agent(selected_agent, task))
    
    async def _execute_task_with_agent(self, agent: BaseAgent, task: Task) -> None:
        """Execute a task with the specified agent."""
        try:
            result = await agent.execute_task(task)
            
            # Move task from active to completed
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            self.completed_tasks.append(task)
            
            logger.info(f"Task {task.id} completed successfully by agent {agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed on agent {agent.agent_id}: {e}")
            
            # Move task from active to failed
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            self.failed_tasks.append(task)
    
    async def _health_monitor(self) -> None:
        """Background task for monitoring agent health."""
        while self.running:
            try:
                for agent in self.agents.values():
                    health = await agent.health_check()
                    if not health["healthy"]:
                        logger.warning(f"Agent {agent.agent_id} health check failed: {health.get('error')}")
                        agent.state = "error"
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_completed_tasks(self) -> None:
        """Background task for cleaning up old completed tasks."""
        while self.running:
            try:
                # Keep only last 1000 completed tasks
                if len(self.completed_tasks) > 1000:
                    self.completed_tasks = self.completed_tasks[-1000:]
                
                if len(self.failed_tasks) > 1000:
                    self.failed_tasks = self.failed_tasks[-1000:]
                
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)
    
    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert a Task object to a dictionary."""
        return {
            "id": task.id,
            "type": task.type,
            "priority": task.priority,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error,
            "result": task.result
        }
