"""
Base agent class and common functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from dataclasses import dataclass
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage
from langchain_community.llms import Ollama
from config import settings, AGENT_STATES

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    type: str
    priority: int
    data: Dict[str, Any]
    created_at: datetime
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.state = AGENT_STATES["idle"]
        self.current_task: Optional[Task] = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.tasks_completed = 0
        self.tasks_failed = 0
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.7
        )
        
        # Agent-specific executor will be set by subclasses
        self.executor: Optional[AgentExecutor] = None
    
    @abstractmethod
    async def setup_agent(self) -> None:
        """Setup the agent with tools and configurations."""
        pass
    
    @abstractmethod
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a specific task assigned to this agent."""
        pass
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task with proper state management and error handling."""
        try:
            self.state = AGENT_STATES["busy"]
            self.current_task = task
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            task.assigned_to = self.agent_id
            
            result = await self.process_task(task)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            self.tasks_completed += 1
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            self.tasks_failed += 1
            self.state = AGENT_STATES["error"]
            raise
            
        finally:
            self.current_task = None
            if self.state != AGENT_STATES["error"]:
                self.state = AGENT_STATES["idle"]
            self.last_activity = datetime.now()
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task type."""
        return task.type in self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state,
            "capabilities": self.capabilities,
            "current_task": self.current_task.id if self.current_task else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "uptime": (datetime.now() - self.created_at).total_seconds(),
            "last_activity": self.last_activity.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the agent."""
        try:
            # Simple test to check if LLM is responding
            response = await asyncio.wait_for(
                self.llm.ainvoke("Hello, are you working?"),
                timeout=10
            )
            return {
                "healthy": True,
                "llm_response": bool(response),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
