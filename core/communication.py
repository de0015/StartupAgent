"""
Communication system for inter-agent messaging and coordination.
"""

import asyncio
import json
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    BROADCAST = "broadcast"
    DIRECT_MESSAGE = "direct_message"

@dataclass
class Message:
    """Represents a message between agents."""
    id: str
    from_agent: str
    to_agent: Optional[str]  # None for broadcast messages
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

class MessageBroker:
    """Central message broker for agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.max_history = 10000
    
    def subscribe(self, message_type: MessageType, callback: Callable) -> str:
        """Subscribe to a specific message type."""
        subscription_id = str(uuid.uuid4())
        
        if message_type.value not in self.subscribers:
            self.subscribers[message_type.value] = []
        
        self.subscribers[message_type.value].append({
            'id': subscription_id,
            'callback': callback
        })
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from message notifications."""
        for message_type, subscribers in self.subscribers.items():
            self.subscribers[message_type] = [
                sub for sub in subscribers if sub['id'] != subscription_id
            ]
    
    async def publish(self, message: Message) -> None:
        """Publish a message to all subscribers."""
        # Add to message history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
        # Notify subscribers
        if message.message_type.value in self.subscribers:
            tasks = []
            for subscriber in self.subscribers[message.message_type.value]:
                tasks.append(self._notify_subscriber(subscriber['callback'], message))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_direct_message(
        self, 
        from_agent: str, 
        to_agent: str, 
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Send a direct message between agents."""
        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.DIRECT_MESSAGE,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        await self.publish(message)
    
    async def broadcast(
        self, 
        from_agent: str, 
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Broadcast a message to all agents."""
        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=None,
            message_type=MessageType.BROADCAST,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        await self.publish(message)
    
    def get_message_history(
        self, 
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[Message]:
        """Get message history with optional filtering."""
        messages = self.message_history
        
        if agent_id:
            messages = [
                msg for msg in messages 
                if msg.from_agent == agent_id or msg.to_agent == agent_id
            ]
        
        if message_type:
            messages = [
                msg for msg in messages 
                if msg.message_type == message_type
            ]
        
        return messages[-limit:]
    
    async def _notify_subscriber(self, callback: Callable, message: Message) -> None:
        """Notify a single subscriber about a message."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            # Log error but don't propagate to avoid breaking other subscribers
            print(f"Error notifying subscriber: {e}")

class AgentCommunicator:
    """Communication interface for individual agents."""
    
    def __init__(self, agent_id: str, message_broker: MessageBroker):
        self.agent_id = agent_id
        self.message_broker = message_broker
        self.subscriptions: List[str] = []
    
    def subscribe_to_messages(self, message_type: MessageType, callback: Callable) -> str:
        """Subscribe to specific message types."""
        subscription_id = self.message_broker.subscribe(message_type, callback)
        self.subscriptions.append(subscription_id)
        return subscription_id
    
    async def send_message(
        self, 
        to_agent: str, 
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Send a message to another agent."""
        await self.message_broker.send_direct_message(
            self.agent_id, to_agent, payload, correlation_id
        )
    
    async def broadcast(
        self, 
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Broadcast a message to all agents."""
        await self.message_broker.broadcast(
            self.agent_id, payload, correlation_id
        )
    
    async def request_task_collaboration(
        self, 
        target_agent: str, 
        task_description: str,
        task_data: Dict[str, Any]
    ) -> str:
        """Request another agent to collaborate on a task."""
        correlation_id = str(uuid.uuid4())
        
        payload = {
            "task_description": task_description,
            "task_data": task_data,
            "requesting_agent": self.agent_id
        }
        
        await self.send_message(target_agent, payload, correlation_id)
        return correlation_id
    
    def cleanup(self) -> None:
        """Clean up subscriptions when agent is destroyed."""
        for subscription_id in self.subscriptions:
            self.message_broker.unsubscribe(subscription_id)
        self.subscriptions.clear()
