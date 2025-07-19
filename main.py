"""
Main application entry point for the Multi-Agent System.
"""

import asyncio
import uvicorn
from typing import Dict, Any
import structlog
from core.agent_manager import AgentManager
from core.performance_monitor import PerformanceMonitor
from dashboard.dashboard import Dashboard

# Import all agent types
from agents.customer_service.customer_service_agent import CustomerServiceAgent
from agents.data_analysis.data_analysis_agent import DataAnalysisAgent
from agents.workflow_automation.workflow_automation_agent import WorkflowAutomationAgent
from agents.integration.integration_agent import IntegrationAgent
from agents.monitoring.monitoring_agent import MonitoringAgent

from config import settings

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class MultiAgentSystem:
    """Main multi-agent system orchestrator."""
    
    def __init__(self):
        self.agent_manager = AgentManager()
        self.performance_monitor = PerformanceMonitor()
        self.dashboard = Dashboard(self.agent_manager, self.performance_monitor)
        self.running = False
        
        # Store agent instances
        self.agents = {}
    
    async def initialize(self) -> None:
        """Initialize the multi-agent system."""
        logger.info("Initializing Multi-Agent System")
        
        try:
            # Start core systems
            await self.agent_manager.start()
            await self.performance_monitor.start()
            
            # Create and register agents
            await self._create_agents()
            
            # Start monitoring and performance tracking
            asyncio.create_task(self._system_health_monitor())
            
            logger.info("Multi-Agent System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _create_agents(self) -> None:
        """Create and register all agent types."""
        logger.info("Creating agents")
        
        try:
            # Create Customer Service Agents
            for i in range(2):  # Create 2 customer service agents
                agent = CustomerServiceAgent(f"cs_agent_{i+1}")
                await agent.setup_agent()
                self.agents[agent.agent_id] = agent
                self.agent_manager.register_agent(agent)
                logger.info(f"Created Customer Service Agent: {agent.agent_id}")
            
            # Create Data Analysis Agent
            agent = DataAnalysisAgent("data_agent_1")
            await agent.setup_agent()
            self.agents[agent.agent_id] = agent
            self.agent_manager.register_agent(agent)
            logger.info(f"Created Data Analysis Agent: {agent.agent_id}")
            
            # Create Workflow Automation Agent
            agent = WorkflowAutomationAgent("workflow_agent_1")
            await agent.setup_agent()
            self.agents[agent.agent_id] = agent
            self.agent_manager.register_agent(agent)
            logger.info(f"Created Workflow Automation Agent: {agent.agent_id}")
            
            # Create Integration Agent
            agent = IntegrationAgent("integration_agent_1")
            await agent.setup_agent()
            self.agents[agent.agent_id] = agent
            self.agent_manager.register_agent(agent)
            logger.info(f"Created Integration Agent: {agent.agent_id}")
            
            # Create Monitoring Agent
            agent = MonitoringAgent("monitoring_agent_1")
            await agent.setup_agent()
            self.agents[agent.agent_id] = agent
            self.agent_manager.register_agent(agent)
            logger.info(f"Created Monitoring Agent: {agent.agent_id}")
            
            logger.info(f"Successfully created {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to create agents: {e}")
            raise
    
    async def start(self) -> None:
        """Start the multi-agent system."""
        logger.info("Starting Multi-Agent System")
        
        try:
            await self.initialize()
            self.running = True
            
            # Submit some sample tasks to demonstrate the system
            await self._submit_sample_tasks()
            
            logger.info("Multi-Agent System is now running")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the multi-agent system."""
        logger.info("Stopping Multi-Agent System")
        
        try:
            self.running = False
            
            # Stop core systems
            await self.agent_manager.stop()
            self.performance_monitor.stop()
            
            logger.info("Multi-Agent System stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    async def _submit_sample_tasks(self) -> None:
        """Submit sample tasks to demonstrate the system."""
        logger.info("Submitting sample tasks")
        
        sample_tasks = [
            {
                "type": "customer_inquiry",
                "data": {
                    "customer_id": "CUST001",
                    "message": "I need help with my recent order",
                    "priority": "medium"
                },
                "priority": "medium"
            },
            {
                "type": "sales_analysis",
                "data": {
                    "period": "monthly",
                    "metrics": ["revenue", "units_sold", "conversion_rate"]
                },
                "priority": "low"
            },
            {
                "type": "document_processing",
                "data": {
                    "document_path": "/data/invoices/invoice_001.pdf",
                    "processing_type": "extract_text",
                    "output_format": "json"
                },
                "priority": "medium"
            },
            {
                "type": "system_monitoring",
                "data": {
                    "systems": ["web_server", "database", "api_gateway"],
                    "interval_seconds": 60
                },
                "priority": "high"
            },
            {
                "type": "api_integration",
                "data": {
                    "api_endpoint": "https://api.example.com/v1",
                    "api_key": "demo_key",
                    "integration_type": "rest"
                },
                "priority": "medium"
            }
        ]
        
        for task_data in sample_tasks:
            task_id = self.agent_manager.submit_task(
                task_data["type"],
                task_data["data"],
                task_data["priority"]
            )
            logger.info(f"Submitted sample task: {task_id} ({task_data['type']})")
    
    async def _system_health_monitor(self) -> None:
        """Monitor overall system health."""
        while self.running:
            try:
                # Perform system health checks
                system_status = self.agent_manager.get_system_status()
                performance_data = self.performance_monitor.get_system_performance()
                
                # Check for any issues
                unhealthy_agents = [
                    agent for agent in system_status["agents"] 
                    if agent["state"] == "error"
                ]
                
                if unhealthy_agents:
                    logger.warning(f"Found {len(unhealthy_agents)} unhealthy agents")
                
                # Check performance metrics
                if performance_data["success_rate"] < 0.9:
                    logger.warning(f"Low success rate: {performance_data['success_rate']:.2%}")
                
                # Broadcast updates to dashboard
                await self.dashboard.broadcast_update({
                    "type": "health_update",
                    "system_status": system_status,
                    "performance_data": performance_data,
                    "timestamp": performance_data["timestamp"].isoformat()
                })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in system health monitor: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_app(self):
        """Get the FastAPI application for running the web server."""
        return self.dashboard.get_app()

# Global system instance
multi_agent_system = MultiAgentSystem()

async def startup_event():
    """FastAPI startup event handler."""
    await multi_agent_system.start()

async def shutdown_event():
    """FastAPI shutdown event handler."""
    await multi_agent_system.stop()

# Add event handlers to the FastAPI app
app = multi_agent_system.get_app()
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

async def main():
    """Main function to run the application."""
    try:
        logger.info("Starting Multi-Agent System Application")
        
        # Start the web server
        config = uvicorn.Config(
            app=app,
            host=settings.api_host,
            port=settings.api_port,
            log_level=settings.log_level.lower(),
            reload=False  # Disable reload in production
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        await multi_agent_system.stop()

if __name__ == "__main__":
    # For development, you can also run with: python main.py
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
