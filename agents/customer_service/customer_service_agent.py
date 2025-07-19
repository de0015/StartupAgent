"""
Customer Service Agents - Handle inquiries, scheduling, and basic support.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from core.base_agent import BaseAgent, Task
import json
import re

class CustomerServiceAgent(BaseAgent):
    """Agent specialized in customer service operations."""
    
    def __init__(self, agent_id: str):
        capabilities = [
            "customer_inquiry",
            "appointment_scheduling", 
            "order_status_check",
            "complaint_handling",
            "product_information",
            "support_ticket_creation"
        ]
        super().__init__(agent_id, "customer_service", capabilities)
        
        # Customer service specific data
        self.knowledge_base = {
            "business_hours": "9:00 AM - 6:00 PM, Monday to Friday",
            "support_email": "support@company.com",
            "support_phone": "1-800-SUPPORT",
            "return_policy": "30-day return policy for all products",
            "shipping_info": "Standard shipping takes 3-5 business days"
        }
        
        self.appointment_slots = self._generate_appointment_slots()
    
    async def setup_agent(self) -> None:
        """Setup the customer service agent with tools and configurations."""
        tools = [
            Tool(
                name="check_appointment_availability",
                description="Check available appointment slots for a given date range",
                func=self._check_appointment_availability
            ),
            Tool(
                name="schedule_appointment",
                description="Schedule an appointment for a customer",
                func=self._schedule_appointment
            ),
            Tool(
                name="get_order_status",
                description="Get the status of a customer order",
                func=self._get_order_status
            ),
            Tool(
                name="create_support_ticket",
                description="Create a support ticket for customer issues",
                func=self._create_support_ticket
            ),
            Tool(
                name="get_product_info",
                description="Get information about products or services",
                func=self._get_product_info
            ),
            Tool(
                name="escalate_to_human",
                description="Escalate complex issues to human support",
                func=self._escalate_to_human
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful customer service representative. Your role is to:
            
            1. Assist customers with inquiries and provide accurate information
            2. Handle appointment scheduling efficiently
            3. Check order statuses and provide updates
            4. Create support tickets for technical issues
            5. Provide product information and recommendations
            6. Escalate complex issues to human agents when necessary
            
            Always be polite, professional, and helpful. If you cannot resolve an issue,
            don't hesitate to escalate to a human agent.
            
            Business Information:
            - Business Hours: {business_hours}
            - Support Email: {support_email}
            - Support Phone: {support_phone}
            - Return Policy: {return_policy}
            - Shipping Info: {shipping_info}
            """.format(**self.knowledge_base)),
            ("user", "{input}"),
            ("assistant", "I'll help you with that. Let me check what I can do for you."),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process customer service related tasks."""
        task_type = task.type
        task_data = task.data
        
        if task_type == "customer_inquiry":
            return await self._handle_customer_inquiry(task_data)
        elif task_type == "appointment_scheduling":
            return await self._handle_appointment_scheduling(task_data)
        elif task_type == "order_status_check":
            return await self._handle_order_status_check(task_data)
        elif task_type == "complaint_handling":
            return await self._handle_complaint(task_data)
        elif task_type == "product_information":
            return await self._handle_product_inquiry(task_data)
        elif task_type == "support_ticket_creation":
            return await self._handle_support_ticket_creation(task_data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _handle_customer_inquiry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general customer inquiries."""
        customer_message = data.get("message", "")
        customer_id = data.get("customer_id", "unknown")
        
        # Use the agent executor to process the inquiry
        response = await self.executor.ainvoke({
            "input": f"Customer {customer_id} asks: {customer_message}"
        })
        
        return {
            "response": response["output"],
            "customer_id": customer_id,
            "handled_by": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "requires_follow_up": self._check_if_follow_up_needed(customer_message)
        }
    
    async def _handle_appointment_scheduling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle appointment scheduling requests."""
        customer_id = data.get("customer_id", "unknown")
        preferred_date = data.get("preferred_date")
        service_type = data.get("service_type", "consultation")
        
        # Check available slots
        available_slots = self._check_appointment_availability(preferred_date)
        
        if available_slots:
            # Schedule the first available slot
            appointment_id = self._schedule_appointment(
                customer_id, available_slots[0], service_type
            )
            
            return {
                "status": "scheduled",
                "appointment_id": appointment_id,
                "scheduled_time": available_slots[0],
                "service_type": service_type,
                "customer_id": customer_id
            }
        else:
            return {
                "status": "no_availability",
                "message": "No available slots for the requested date",
                "alternative_dates": self._suggest_alternative_dates(preferred_date)
            }
    
    async def _handle_order_status_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle order status inquiries."""
        order_id = data.get("order_id")
        customer_id = data.get("customer_id")
        
        order_status = self._get_order_status(order_id)
        
        return {
            "order_id": order_id,
            "status": order_status,
            "customer_id": customer_id,
            "checked_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_complaint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer complaints."""
        customer_id = data.get("customer_id")
        complaint = data.get("complaint")
        severity = data.get("severity", "medium")
        
        # Create support ticket for complaint
        ticket_id = self._create_support_ticket(
            customer_id, f"Complaint: {complaint}", severity
        )
        
        # Determine if escalation is needed
        escalate = severity in ["high", "critical"] or self._contains_escalation_keywords(complaint)
        
        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "severity": severity,
            "escalated": escalate,
            "handled_by": self.agent_id
        }
        
        if escalate:
            result["escalation_info"] = self._escalate_to_human(ticket_id, complaint)
        
        return result
    
    async def _handle_product_inquiry(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle product information requests."""
        product_name = data.get("product_name")
        inquiry_type = data.get("inquiry_type", "general")
        
        product_info = self._get_product_info(product_name)
        
        return {
            "product_name": product_name,
            "inquiry_type": inquiry_type,
            "product_info": product_info,
            "provided_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_support_ticket_creation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle support ticket creation."""
        customer_id = data.get("customer_id")
        issue_description = data.get("issue_description")
        priority = data.get("priority", "medium")
        
        ticket_id = self._create_support_ticket(customer_id, issue_description, priority)
        
        return {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "priority": priority,
            "created_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_appointment_slots(self) -> List[str]:
        """Generate available appointment slots for the next 30 days."""
        slots = []
        start_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        
        for day in range(30):  # Next 30 days
            current_date = start_date + timedelta(days=day)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Generate hourly slots from 9 AM to 5 PM
            for hour in range(9, 17):
                slot_time = current_date.replace(hour=hour)
                slots.append(slot_time.isoformat())
        
        return slots
    
    def _check_appointment_availability(self, date: str) -> List[str]:
        """Check available appointment slots for a specific date."""
        try:
            target_date = datetime.fromisoformat(date.replace('Z', '+00:00')).date()
        except:
            target_date = datetime.now().date()
        
        available = [
            slot for slot in self.appointment_slots
            if datetime.fromisoformat(slot).date() == target_date
        ]
        
        return available[:5]  # Return up to 5 available slots
    
    def _schedule_appointment(self, customer_id: str, slot: str, service_type: str) -> str:
        """Schedule an appointment."""
        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d')}-{len(self.appointment_slots)}"
        
        # Remove the slot from available slots
        if slot in self.appointment_slots:
            self.appointment_slots.remove(slot)
        
        return appointment_id
    
    def _get_order_status(self, order_id: str) -> str:
        """Get order status (mock implementation)."""
        # This would normally connect to an order management system
        mock_statuses = ["processing", "shipped", "delivered", "cancelled"]
        import random
        return random.choice(mock_statuses)
    
    def _create_support_ticket(self, customer_id: str, issue: str, priority: str) -> str:
        """Create a support ticket."""
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # This would normally save to a ticketing system
        return ticket_id
    
    def _get_product_info(self, product_name: str) -> Dict[str, Any]:
        """Get product information (mock implementation)."""
        # This would normally query a product database
        return {
            "name": product_name,
            "description": f"Information about {product_name}",
            "price": "$99.99",
            "availability": "In stock",
            "features": ["Feature 1", "Feature 2", "Feature 3"]
        }
    
    def _escalate_to_human(self, ticket_id: str, issue: str) -> Dict[str, Any]:
        """Escalate issue to human agent."""
        return {
            "escalated_to": "human_support_team",
            "escalation_time": datetime.now().isoformat(),
            "reason": "Complex issue requiring human intervention",
            "ticket_id": ticket_id
        }
    
    def _suggest_alternative_dates(self, preferred_date: str) -> List[str]:
        """Suggest alternative appointment dates."""
        try:
            preferred = datetime.fromisoformat(preferred_date.replace('Z', '+00:00'))
        except:
            preferred = datetime.now()
        
        alternatives = []
        for i in range(1, 8):  # Next 7 days
            alt_date = preferred + timedelta(days=i)
            if alt_date.weekday() < 5:  # Weekdays only
                alternatives.append(alt_date.date().isoformat())
        
        return alternatives[:3]  # Return 3 alternatives
    
    def _check_if_follow_up_needed(self, message: str) -> bool:
        """Check if the customer message requires follow-up."""
        follow_up_keywords = [
            "urgent", "asap", "immediately", "emergency", 
            "not working", "broken", "error", "problem"
        ]
        
        return any(keyword in message.lower() for keyword in follow_up_keywords)
    
    def _contains_escalation_keywords(self, complaint: str) -> bool:
        """Check if complaint contains keywords that require escalation."""
        escalation_keywords = [
            "lawsuit", "lawyer", "legal", "sue", "fraud",
            "terrible", "worst", "hate", "disgusted", "outraged"
        ]
        
        return any(keyword in complaint.lower() for keyword in escalation_keywords)
