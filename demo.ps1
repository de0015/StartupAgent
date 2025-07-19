# Multi-Agent System Demo Script

Write-Host "🎬 Multi-Agent System Demo" -ForegroundColor Cyan
Write-Host "This script will demonstrate the capabilities of the multi-agent system" -ForegroundColor White
Write-Host ""

# Define demo tasks
$demoTasks = @(
    @{
        name = "Customer Service Inquiry"
        type = "customer_inquiry"
        data = @{
            customer_id = "DEMO001"
            message = "I need help with my recent order. It hasn't arrived yet."
            priority = "high"
        }
        priority = "high"
        description = "Simulate a customer service request"
    },
    @{
        name = "Sales Data Analysis"
        type = "sales_analysis"
        data = @{
            period = "monthly"
            metrics = @("revenue", "units_sold", "conversion_rate")
        }
        priority = "medium"
        description = "Analyze monthly sales performance"
    },
    @{
        name = "Document Processing"
        type = "document_processing"
        data = @{
            document_path = "/demo/sample_invoice.pdf"
            processing_type = "extract_text"
            output_format = "json"
        }
        priority = "medium"
        description = "Extract data from a sample invoice"
    },
    @{
        name = "System Health Check"
        type = "system_monitoring"
        data = @{
            systems = @("web_server", "database", "api_gateway")
            interval_seconds = 30
        }
        priority = "high"
        description = "Monitor system health and performance"
    },
    @{
        name = "API Integration"
        type = "api_integration"
        data = @{
            api_endpoint = "https://jsonplaceholder.typicode.com/posts"
            integration_type = "rest"
        }
        priority = "low"
        description = "Connect to external API service"
    },
    @{
        name = "Workflow Automation"
        type = "workflow_orchestration"
        data = @{
            workflow_definition = @{
                id = "demo_workflow"
                steps = @(
                    @{ type = "data_validation"; name = "Validate Input" },
                    @{ type = "data_transformation"; name = "Transform Data" },
                    @{ type = "notification"; name = "Send Completion Notice" }
                )
            }
            input_data = @{
                sample_data = "Demo workflow data"
            }
        }
        priority = "medium"
        description = "Execute a multi-step workflow"
    }
)

Write-Host "Available Demo Tasks:" -ForegroundColor Yellow
for ($i = 0; $i -lt $demoTasks.Count; $i++) {
    $task = $demoTasks[$i]
    Write-Host "  $($i + 1). $($task.name) - $($task.description)" -ForegroundColor White
}
Write-Host ""

# Function to submit a task via API
function Submit-Task {
    param (
        [string]$Type,
        [hashtable]$Data,
        [string]$Priority = "medium"
    )
    
    $body = @{
        type = $Type
        data = $Data
        priority = $Priority
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/tasks" -Method POST -Body $body -ContentType "application/json"
        return $response
    } catch {
        Write-Host "❌ Failed to submit task: $_" -ForegroundColor Red
        return $null
    }
}

# Function to check task status
function Get-TaskStatus {
    param ([string]$TaskId)
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/tasks/$TaskId" -Method GET
        return $response
    } catch {
        return $null
    }
}

# Function to get system status
function Get-SystemStatus {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/system/status" -Method GET
        return $response
    } catch {
        Write-Host "❌ Failed to get system status. Is the system running?" -ForegroundColor Red
        return $null
    }
}

# Main demo menu
do {
    Write-Host ""
    Write-Host "Multi-Agent System Demo Menu:" -ForegroundColor Cyan
    Write-Host "1. Submit individual demo task" -ForegroundColor White
    Write-Host "2. Submit all demo tasks" -ForegroundColor White
    Write-Host "3. Check system status" -ForegroundColor White
    Write-Host "4. Monitor task progress" -ForegroundColor White
    Write-Host "5. Open dashboard in browser" -ForegroundColor White
    Write-Host "6. Exit demo" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "Enter your choice (1-6)"
    
    switch ($choice) {
        "1" {
            Write-Host ""
            Write-Host "Select a task to submit:" -ForegroundColor Yellow
            for ($i = 0; $i -lt $demoTasks.Count; $i++) {
                $task = $demoTasks[$i]
                Write-Host "  $($i + 1). $($task.name)" -ForegroundColor White
            }
            
            $taskChoice = Read-Host "Enter task number (1-$($demoTasks.Count))"
            
            if ($taskChoice -match '^\d+$' -and [int]$taskChoice -ge 1 -and [int]$taskChoice -le $demoTasks.Count) {
                $selectedTask = $demoTasks[[int]$taskChoice - 1]
                Write-Host "Submitting: $($selectedTask.name)..." -ForegroundColor Yellow
                
                $result = Submit-Task -Type $selectedTask.type -Data $selectedTask.data -Priority $selectedTask.priority
                
                if ($result) {
                    Write-Host "✓ Task submitted successfully!" -ForegroundColor Green
                    Write-Host "  Task ID: $($result.task_id)" -ForegroundColor White
                    Write-Host "  Status: $($result.status)" -ForegroundColor White
                } else {
                    Write-Host "❌ Failed to submit task" -ForegroundColor Red
                }
            } else {
                Write-Host "Invalid task number" -ForegroundColor Red
            }
        }
        
        "2" {
            Write-Host "Submitting all demo tasks..." -ForegroundColor Yellow
            $submittedTasks = @()
            
            foreach ($task in $demoTasks) {
                Write-Host "Submitting: $($task.name)..." -ForegroundColor White
                $result = Submit-Task -Type $task.type -Data $task.data -Priority $task.priority
                
                if ($result) {
                    $submittedTasks += @{
                        name = $task.name
                        id = $result.task_id
                    }
                    Write-Host "  ✓ $($task.name) submitted (ID: $($result.task_id))" -ForegroundColor Green
                } else {
                    Write-Host "  ❌ Failed to submit $($task.name)" -ForegroundColor Red
                }
                
                Start-Sleep -Seconds 1
            }
            
            Write-Host ""
            Write-Host "All tasks submitted! Monitor progress in the dashboard." -ForegroundColor Green
        }
        
        "3" {
            Write-Host "Getting system status..." -ForegroundColor Yellow
            $status = Get-SystemStatus
            
            if ($status) {
                Write-Host ""
                Write-Host "System Status:" -ForegroundColor Cyan
                Write-Host "  Total Agents: $($status.system.total_agents)" -ForegroundColor White
                Write-Host "  Active Tasks: $($status.system.active_tasks)" -ForegroundColor White
                Write-Host "  Pending Tasks: $($status.system.pending_tasks)" -ForegroundColor White
                Write-Host "  Completed Tasks: $($status.system.completed_tasks)" -ForegroundColor White
                Write-Host "  Failed Tasks: $($status.system.failed_tasks)" -ForegroundColor White
                Write-Host ""
                Write-Host "Performance:" -ForegroundColor Cyan
                Write-Host "  Success Rate: $([math]::Round($status.performance.success_rate * 100, 1))%" -ForegroundColor White
                Write-Host "  Avg Task Duration: $([math]::Round($status.performance.avg_task_duration, 2))s" -ForegroundColor White
                Write-Host "  Throughput: $([math]::Round($status.performance.throughput_per_minute, 1)) tasks/min" -ForegroundColor White
                Write-Host "  System Uptime: $([math]::Round($status.performance.system_uptime / 3600, 1))h" -ForegroundColor White
            }
        }
        
        "4" {
            Write-Host "Monitoring task progress (Press Ctrl+C to stop)..." -ForegroundColor Yellow
            
            try {
                while ($true) {
                    Clear-Host
                    Write-Host "Multi-Agent System - Live Task Monitor" -ForegroundColor Cyan
                    Write-Host "===============================================" -ForegroundColor Cyan
                    
                    $status = Get-SystemStatus
                    if ($status) {
                        Write-Host "Active Tasks: $($status.system.active_tasks) | " -NoNewline -ForegroundColor Green
                        Write-Host "Pending: $($status.system.pending_tasks) | " -NoNewline -ForegroundColor Yellow
                        Write-Host "Completed: $($status.system.completed_tasks) | " -NoNewline -ForegroundColor Blue
                        Write-Host "Failed: $($status.system.failed_tasks)" -ForegroundColor Red
                        Write-Host ""
                        
                        Write-Host "Agent States:" -ForegroundColor White
                        foreach ($agent in $status.system.agents) {
                            $stateColor = switch ($agent.state) {
                                "idle" { "White" }
                                "busy" { "Yellow" }
                                "error" { "Red" }
                                default { "Gray" }
                            }
                            Write-Host "  $($agent.agent_id): $($agent.state) ($($agent.tasks_completed) completed)" -ForegroundColor $stateColor
                        }
                    }
                    
                    Start-Sleep -Seconds 2
                }
            } catch {
                Write-Host ""
                Write-Host "Monitoring stopped." -ForegroundColor Yellow
            }
        }
        
        "5" {
            Write-Host "Opening dashboard in browser..." -ForegroundColor Yellow
            Start-Process "http://localhost:8000"
        }
        
        "6" {
            Write-Host "Exiting demo..." -ForegroundColor Yellow
            break
        }
        
        default {
            Write-Host "Invalid choice. Please enter 1-6." -ForegroundColor Red
        }
    }
} while ($true)

Write-Host ""
Write-Host "👋 Demo completed. Thank you for exploring the Multi-Agent System!" -ForegroundColor Green
