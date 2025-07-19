"""
Web Dashboard for the Multi-Agent System
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from core.agent_manager import AgentManager
from core.performance_monitor import PerformanceMonitor

class Dashboard:
    """Web dashboard for monitoring and controlling the multi-agent system."""
    
    def __init__(self, agent_manager: AgentManager, performance_monitor: PerformanceMonitor):
        self.app = FastAPI(title="Multi-Agent System Dashboard")
        self.agent_manager = agent_manager
        self.performance_monitor = performance_monitor
        self.templates = Jinja2Templates(directory="dashboard/templates")
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        self._setup_routes()
        self._setup_static_files()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            system_status = self.agent_manager.get_system_status()
            performance_data = self.performance_monitor.get_system_performance()
            
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "system_status": system_status,
                "performance_data": performance_data,
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.get("/api/agents")
        async def get_agents():
            """Get all agents status."""
            agents = []
            for agent_id, agent in self.agent_manager.agents.items():
                agent_status = agent.get_status()
                agent_performance = self.performance_monitor.get_agent_performance(agent_id)
                agents.append({
                    **agent_status,
                    "performance": agent_performance
                })
            return {"agents": agents}
        
        @self.app.get("/api/agents/{agent_id}")
        async def get_agent_details(agent_id: str):
            """Get detailed information about a specific agent."""
            agent_status = self.agent_manager.get_agent_status(agent_id)
            if agent_status:
                agent_performance = self.performance_monitor.get_agent_performance(agent_id)
                return {
                    "agent": agent_status,
                    "performance": agent_performance
                }
            return {"error": "Agent not found"}, 404
        
        @self.app.get("/api/tasks")
        async def get_tasks():
            """Get all tasks status."""
            return {
                "pending_tasks": self.agent_manager.task_queue.size(),
                "active_tasks": len(self.agent_manager.active_tasks),
                "completed_tasks": len(self.agent_manager.completed_tasks),
                "failed_tasks": len(self.agent_manager.failed_tasks),
                "active_task_details": [
                    self.agent_manager._task_to_dict(task) 
                    for task in self.agent_manager.active_tasks.values()
                ],
                "recent_completed": [
                    self.agent_manager._task_to_dict(task) 
                    for task in self.agent_manager.completed_tasks[-10:]
                ]
            }
        
        @self.app.post("/api/tasks")
        async def submit_task(task_data: Dict[str, Any]):
            """Submit a new task to the system."""
            task_type = task_data.get("type")
            data = task_data.get("data", {})
            priority = task_data.get("priority", "medium")
            
            if not task_type:
                return {"error": "Task type is required"}, 400
            
            task_id = self.agent_manager.submit_task(task_type, data, priority)
            return {"task_id": task_id, "status": "submitted"}
        
        @self.app.get("/api/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """Get status of a specific task."""
            task_status = self.agent_manager.get_task_status(task_id)
            if task_status:
                return {"task": task_status}
            return {"error": "Task not found"}, 404
        
        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get overall system status."""
            system_status = self.agent_manager.get_system_status()
            performance_data = self.performance_monitor.get_system_performance()
            task_performance = self.performance_monitor.get_task_type_performance()
            
            return {
                "system": system_status,
                "performance": performance_data,
                "task_performance": task_performance,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get performance metrics."""
            return {
                "system_performance": self.performance_monitor.get_system_performance(),
                "task_type_performance": self.performance_monitor.get_task_type_performance(),
                "agent_performance": {
                    agent_id: self.performance_monitor.get_agent_performance(agent_id)
                    for agent_id in self.agent_manager.agents.keys()
                }
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                    system_status = self.agent_manager.get_system_status()
                    performance_data = self.performance_monitor.get_system_performance()
                    
                    update_data = {
                        "type": "system_update",
                        "system_status": system_status,
                        "performance_data": performance_data,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await websocket.send_text(json.dumps(update_data))
                    
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
        
        @self.app.get("/agents", response_class=HTMLResponse)
        async def agents_page(request: Request):
            """Agents management page."""
            agents_data = []
            for agent_id, agent in self.agent_manager.agents.items():
                agent_status = agent.get_status()
                agent_performance = self.performance_monitor.get_agent_performance(agent_id)
                agents_data.append({
                    **agent_status,
                    "performance": agent_performance
                })
            
            return self.templates.TemplateResponse("agents.html", {
                "request": request,
                "agents": agents_data
            })
        
        @self.app.get("/tasks", response_class=HTMLResponse)
        async def tasks_page(request: Request):
            """Tasks management page."""
            tasks_data = {
                "pending": self.agent_manager.task_queue.size(),
                "active": len(self.agent_manager.active_tasks),
                "completed": len(self.agent_manager.completed_tasks),
                "failed": len(self.agent_manager.failed_tasks),
                "active_tasks": [
                    self.agent_manager._task_to_dict(task) 
                    for task in self.agent_manager.active_tasks.values()
                ],
                "recent_completed": [
                    self.agent_manager._task_to_dict(task) 
                    for task in self.agent_manager.completed_tasks[-20:]
                ]
            }
            
            return self.templates.TemplateResponse("tasks.html", {
                "request": request,
                "tasks": tasks_data
            })
        
        @self.app.get("/metrics", response_class=HTMLResponse)
        async def metrics_page(request: Request):
            """Performance metrics page."""
            metrics_data = {
                "system_performance": self.performance_monitor.get_system_performance(),
                "task_type_performance": self.performance_monitor.get_task_type_performance(),
                "agent_performance": {
                    agent_id: self.performance_monitor.get_agent_performance(agent_id)
                    for agent_id in self.agent_manager.agents.keys()
                }
            }
            
            return self.templates.TemplateResponse("metrics.html", {
                "request": request,
                "metrics": metrics_data
            })
    
    def _setup_static_files(self):
        """Setup static file serving."""
        try:
            self.app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
        except:
            # Create basic static directory if it doesn't exist
            import os
            os.makedirs("dashboard/static", exist_ok=True)
            self.app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
    
    async def broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients."""
        if self.active_connections:
            message = json.dumps(update_data)
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"Error broadcasting to WebSocket: {e}")
                    self.active_connections.remove(connection)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
