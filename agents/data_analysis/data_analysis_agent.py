"""
Data Analysis Agents - Process business metrics, identify patterns, generate reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from core.base_agent import BaseAgent, Task
import json
import io
import base64

class DataAnalysisAgent(BaseAgent):
    """Agent specialized in data analysis and business intelligence."""
    
    def __init__(self, agent_id: str):
        capabilities = [
            "sales_analysis",
            "customer_behavior_analysis",
            "performance_metrics",
            "trend_analysis",
            "report_generation",
            "data_visualization",
            "predictive_modeling",
            "anomaly_detection"
        ]
        super().__init__(agent_id, "data_analysis", capabilities)
        
        # Sample data for demonstration
        self.sample_data = self._generate_sample_data()
    
    async def setup_agent(self) -> None:
        """Setup the data analysis agent with tools and configurations."""
        tools = [
            Tool(
                name="analyze_sales_data",
                description="Analyze sales data and generate insights",
                func=self._analyze_sales_data
            ),
            Tool(
                name="customer_segmentation",
                description="Perform customer segmentation analysis",
                func=self._customer_segmentation
            ),
            Tool(
                name="trend_analysis",
                description="Analyze trends in business metrics",
                func=self._trend_analysis
            ),
            Tool(
                name="generate_visualization",
                description="Create data visualizations and charts",
                func=self._generate_visualization
            ),
            Tool(
                name="detect_anomalies",
                description="Detect anomalies in data patterns",
                func=self._detect_anomalies
            ),
            Tool(
                name="calculate_kpis",
                description="Calculate key performance indicators",
                func=self._calculate_kpis
            ),
            Tool(
                name="forecast_metrics",
                description="Generate forecasts for business metrics",
                func=self._forecast_metrics
            ),
            Tool(
                name="correlation_analysis",
                description="Analyze correlations between variables",
                func=self._correlation_analysis
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst and business intelligence expert. Your role is to:
            
            1. Analyze business data to extract meaningful insights
            2. Identify trends and patterns in customer behavior
            3. Generate comprehensive reports with actionable recommendations
            4. Create visualizations to communicate findings effectively
            5. Detect anomalies and unusual patterns in data
            6. Forecast future trends and metrics
            7. Calculate and monitor key performance indicators
            
            Always provide data-driven insights with clear explanations and recommendations.
            Use statistical methods and visualization tools to support your analysis.
            """),
            ("user", "{input}"),
            ("assistant", "I'll analyze the data and provide you with insights. Let me start by examining the available data."),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process data analysis related tasks."""
        task_type = task.type
        task_data = task.data
        
        if task_type == "sales_analysis":
            return await self._handle_sales_analysis(task_data)
        elif task_type == "customer_behavior_analysis":
            return await self._handle_customer_behavior_analysis(task_data)
        elif task_type == "performance_metrics":
            return await self._handle_performance_metrics(task_data)
        elif task_type == "trend_analysis":
            return await self._handle_trend_analysis(task_data)
        elif task_type == "report_generation":
            return await self._handle_report_generation(task_data)
        elif task_type == "data_visualization":
            return await self._handle_data_visualization(task_data)
        elif task_type == "predictive_modeling":
            return await self._handle_predictive_modeling(task_data)
        elif task_type == "anomaly_detection":
            return await self._handle_anomaly_detection(task_data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _handle_sales_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sales data analysis."""
        period = data.get("period", "monthly")
        metrics = data.get("metrics", ["revenue", "units_sold", "conversion_rate"])
        
        sales_insights = self._analyze_sales_data(period, metrics)
        
        return {
            "analysis_type": "sales_analysis",
            "period": period,
            "metrics_analyzed": metrics,
            "insights": sales_insights,
            "recommendations": self._generate_sales_recommendations(sales_insights),
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_customer_behavior_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle customer behavior analysis."""
        segment_criteria = data.get("segment_criteria", ["purchase_frequency", "total_spent"])
        
        segmentation_results = self._customer_segmentation(segment_criteria)
        
        return {
            "analysis_type": "customer_behavior",
            "segmentation_criteria": segment_criteria,
            "segments": segmentation_results,
            "insights": self._generate_customer_insights(segmentation_results),
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance metrics calculation."""
        kpi_types = data.get("kpi_types", ["conversion_rate", "avg_order_value", "customer_lifetime_value"])
        
        kpis = self._calculate_kpis(kpi_types)
        
        return {
            "analysis_type": "performance_metrics",
            "kpi_types": kpi_types,
            "kpis": kpis,
            "performance_status": self._assess_performance(kpis),
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trend analysis."""
        metric = data.get("metric", "revenue")
        timeframe = data.get("timeframe", "6_months")
        
        trend_analysis = self._trend_analysis(metric, timeframe)
        
        return {
            "analysis_type": "trend_analysis",
            "metric": metric,
            "timeframe": timeframe,
            "trend_data": trend_analysis,
            "forecast": self._forecast_metrics(metric, 3),  # 3 months forecast
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_report_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive report generation."""
        report_type = data.get("report_type", "executive_summary")
        include_visualizations = data.get("include_visualizations", True)
        
        report = self._generate_comprehensive_report(report_type, include_visualizations)
        
        return {
            "report_type": report_type,
            "report_content": report,
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_data_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data visualization requests."""
        chart_type = data.get("chart_type", "bar")
        data_source = data.get("data_source", "sales")
        
        visualization = self._generate_visualization(chart_type, data_source)
        
        return {
            "visualization_type": chart_type,
            "data_source": data_source,
            "chart_data": visualization,
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_predictive_modeling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle predictive modeling tasks."""
        target_metric = data.get("target_metric", "revenue")
        prediction_horizon = data.get("prediction_horizon", 3)  # months
        
        predictions = self._forecast_metrics(target_metric, prediction_horizon)
        
        return {
            "target_metric": target_metric,
            "prediction_horizon": prediction_horizon,
            "predictions": predictions,
            "model_accuracy": self._calculate_model_accuracy(),
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_anomaly_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anomaly detection tasks."""
        metric = data.get("metric", "daily_sales")
        sensitivity = data.get("sensitivity", "medium")
        
        anomalies = self._detect_anomalies(metric, sensitivity)
        
        return {
            "metric": metric,
            "sensitivity": sensitivity,
            "anomalies_detected": anomalies,
            "recommendations": self._generate_anomaly_recommendations(anomalies),
            "generated_by": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample business data for analysis."""
        # Sales data
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        sales_data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(5000, 1000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 500,
            'units_sold': np.random.poisson(50, len(dates)),
            'visitors': np.random.normal(200, 50, len(dates)),
            'conversion_rate': np.random.beta(2, 20, len(dates))
        })
        
        # Customer data
        customer_data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'total_spent': np.random.lognormal(6, 1, 1000),
            'purchase_frequency': np.random.poisson(3, 1000),
            'last_purchase_days': np.random.exponential(30, 1000),
            'segment': np.random.choice(['High Value', 'Medium Value', 'Low Value'], 1000, p=[0.2, 0.3, 0.5])
        })
        
        return {
            'sales': sales_data,
            'customers': customer_data
        }
    
    def _analyze_sales_data(self, period: str, metrics: List[str]) -> Dict[str, Any]:
        """Analyze sales data for specified period and metrics."""
        sales_df = self.sample_data['sales']
        
        if period == "monthly":
            grouped = sales_df.groupby(sales_df['date'].dt.to_period('M'))
        elif period == "weekly":
            grouped = sales_df.groupby(sales_df['date'].dt.to_period('W'))
        else:  # daily
            grouped = sales_df.groupby(sales_df['date'].dt.date)
        
        analysis = {}
        for metric in metrics:
            if metric in sales_df.columns:
                metric_data = grouped[metric].agg(['sum', 'mean', 'std'])
                analysis[metric] = {
                    'total': float(metric_data['sum'].sum()),
                    'average': float(metric_data['mean'].mean()),
                    'growth_rate': float((metric_data['sum'].iloc[-1] - metric_data['sum'].iloc[0]) / metric_data['sum'].iloc[0] * 100),
                    'volatility': float(metric_data['std'].mean())
                }
        
        return analysis
    
    def _customer_segmentation(self, criteria: List[str]) -> Dict[str, Any]:
        """Perform customer segmentation analysis."""
        customer_df = self.sample_data['customers']
        
        # Simple segmentation based on total_spent and purchase_frequency
        high_value = customer_df[
            (customer_df['total_spent'] > customer_df['total_spent'].quantile(0.8)) &
            (customer_df['purchase_frequency'] > customer_df['purchase_frequency'].quantile(0.6))
        ]
        
        medium_value = customer_df[
            (customer_df['total_spent'] > customer_df['total_spent'].quantile(0.4)) &
            (customer_df['total_spent'] <= customer_df['total_spent'].quantile(0.8))
        ]
        
        low_value = customer_df[
            customer_df['total_spent'] <= customer_df['total_spent'].quantile(0.4)
        ]
        
        return {
            'high_value': {
                'count': len(high_value),
                'avg_spent': float(high_value['total_spent'].mean()),
                'avg_frequency': float(high_value['purchase_frequency'].mean())
            },
            'medium_value': {
                'count': len(medium_value),
                'avg_spent': float(medium_value['total_spent'].mean()),
                'avg_frequency': float(medium_value['purchase_frequency'].mean())
            },
            'low_value': {
                'count': len(low_value),
                'avg_spent': float(low_value['total_spent'].mean()),
                'avg_frequency': float(low_value['purchase_frequency'].mean())
            }
        }
    
    def _trend_analysis(self, metric: str, timeframe: str) -> Dict[str, Any]:
        """Analyze trends in specified metric over timeframe."""
        sales_df = self.sample_data['sales']
        
        if timeframe == "6_months":
            cutoff_date = datetime.now() - timedelta(days=180)
        elif timeframe == "1_year":
            cutoff_date = datetime.now() - timedelta(days=365)
        else:  # 3_months
            cutoff_date = datetime.now() - timedelta(days=90)
        
        filtered_df = sales_df[sales_df['date'] >= cutoff_date]
        
        if metric in filtered_df.columns:
            values = filtered_df[metric].values
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
            
            return {
                'metric': metric,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'trend_strength': abs(float(trend_slope)),
                'average_value': float(np.mean(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'data_points': len(values)
            }
        
        return {'error': f'Metric {metric} not found'}
    
    def _generate_visualization(self, chart_type: str, data_source: str) -> Dict[str, Any]:
        """Generate data visualizations."""
        if data_source == "sales":
            df = self.sample_data['sales'].tail(30)  # Last 30 days
            
            if chart_type == "line":
                fig = px.line(df, x='date', y='revenue', title='Revenue Trend (Last 30 Days)')
            elif chart_type == "bar":
                monthly_data = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
                fig = px.bar(x=monthly_data.index.astype(str), y=monthly_data.values, title='Monthly Revenue')
            else:  # scatter
                fig = px.scatter(df, x='visitors', y='revenue', title='Revenue vs Visitors')
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return {
                'chart_type': chart_type,
                'title': fig.layout.title.text,
                'image_data': img_str,
                'data_points': len(df)
            }
        
        return {'error': f'Data source {data_source} not available'}
    
    def _detect_anomalies(self, metric: str, sensitivity: str) -> List[Dict[str, Any]]:
        """Detect anomalies in the specified metric."""
        sales_df = self.sample_data['sales']
        
        if metric in sales_df.columns:
            values = sales_df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Set threshold based on sensitivity
            threshold_multiplier = {'low': 3, 'medium': 2, 'high': 1.5}[sensitivity]
            threshold = threshold_multiplier * std_val
            
            anomalies = []
            for i, value in enumerate(values):
                if abs(value - mean_val) > threshold:
                    anomalies.append({
                        'index': i,
                        'date': sales_df.iloc[i]['date'].isoformat(),
                        'value': float(value),
                        'deviation': float(abs(value - mean_val)),
                        'severity': 'high' if abs(value - mean_val) > 2 * threshold else 'medium'
                    })
            
            return anomalies
        
        return []
    
    def _calculate_kpis(self, kpi_types: List[str]) -> Dict[str, float]:
        """Calculate key performance indicators."""
        sales_df = self.sample_data['sales']
        customer_df = self.sample_data['customers']
        
        kpis = {}
        
        for kpi in kpi_types:
            if kpi == "conversion_rate":
                kpis[kpi] = float(sales_df['conversion_rate'].mean())
            elif kpi == "avg_order_value":
                kpis[kpi] = float(sales_df['revenue'].sum() / sales_df['units_sold'].sum())
            elif kpi == "customer_lifetime_value":
                kpis[kpi] = float(customer_df['total_spent'].mean())
            elif kpi == "churn_rate":
                # Mock calculation
                recent_customers = len(customer_df[customer_df['last_purchase_days'] <= 30])
                total_customers = len(customer_df)
                kpis[kpi] = float(1 - (recent_customers / total_customers))
        
        return kpis
    
    def _forecast_metrics(self, metric: str, months: int) -> Dict[str, Any]:
        """Generate forecasts for business metrics."""
        sales_df = self.sample_data['sales']
        
        if metric in sales_df.columns:
            values = sales_df[metric].values
            
            # Simple linear trend forecast
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            
            # Generate future predictions
            future_x = np.arange(len(values), len(values) + months * 30)  # Approximate days
            predictions = np.polyval(coeffs, future_x)
            
            return {
                'metric': metric,
                'forecast_period_months': months,
                'predictions': [float(p) for p in predictions[::30][:months]],  # Monthly values
                'trend_slope': float(coeffs[0]),
                'confidence_interval': 0.8  # Mock confidence
            }
        
        return {'error': f'Metric {metric} not found'}
    
    def _correlation_analysis(self, variables: List[str]) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        sales_df = self.sample_data['sales']
        
        available_vars = [var for var in variables if var in sales_df.columns]
        if len(available_vars) < 2:
            return {'error': 'Need at least 2 valid variables for correlation analysis'}
        
        correlation_matrix = sales_df[available_vars].corr()
        
        return {
            'variables': available_vars,
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find strong correlations in the correlation matrix."""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'very strong' if abs(corr_value) > 0.9 else 'strong'
                    })
        
        return strong_corrs
    
    def _generate_sales_recommendations(self, sales_insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sales analysis."""
        recommendations = []
        
        for metric, data in sales_insights.items():
            if data.get('growth_rate', 0) < 0:
                recommendations.append(f"Address declining {metric} - consider marketing campaigns or product improvements")
            elif data.get('volatility', 0) > data.get('average', 0) * 0.5:
                recommendations.append(f"High volatility in {metric} - investigate factors causing fluctuations")
        
        return recommendations
    
    def _generate_customer_insights(self, segmentation: Dict[str, Any]) -> List[str]:
        """Generate insights from customer segmentation."""
        insights = []
        
        total_customers = sum(segment['count'] for segment in segmentation.values())
        high_value_percent = (segmentation['high_value']['count'] / total_customers) * 100
        
        if high_value_percent < 20:
            insights.append("Low percentage of high-value customers - consider loyalty programs")
        
        if segmentation['low_value']['avg_frequency'] < 2:
            insights.append("Low-value customers have infrequent purchases - target with engagement campaigns")
        
        return insights
    
    def _assess_performance(self, kpis: Dict[str, float]) -> str:
        """Assess overall performance based on KPIs."""
        # Simple performance assessment logic
        conversion_rate = kpis.get('conversion_rate', 0)
        churn_rate = kpis.get('churn_rate', 0)
        
        if conversion_rate > 0.05 and churn_rate < 0.1:
            return "Excellent"
        elif conversion_rate > 0.03 and churn_rate < 0.2:
            return "Good"
        elif conversion_rate > 0.02:
            return "Average"
        else:
            return "Needs Improvement"
    
    def _generate_comprehensive_report(self, report_type: str, include_viz: bool) -> Dict[str, Any]:
        """Generate a comprehensive business report."""
        sales_analysis = self._analyze_sales_data("monthly", ["revenue", "units_sold"])
        customer_segments = self._customer_segmentation(["total_spent", "purchase_frequency"])
        kpis = self._calculate_kpis(["conversion_rate", "avg_order_value", "customer_lifetime_value"])
        
        report = {
            'report_type': report_type,
            'summary': {
                'total_revenue': sales_analysis.get('revenue', {}).get('total', 0),
                'total_units_sold': sales_analysis.get('units_sold', {}).get('total', 0),
                'avg_conversion_rate': kpis.get('conversion_rate', 0),
                'performance_status': self._assess_performance(kpis)
            },
            'sales_analysis': sales_analysis,
            'customer_segmentation': customer_segments,
            'key_performance_indicators': kpis,
            'recommendations': [
                "Focus on high-value customer retention",
                "Improve conversion rate through A/B testing",
                "Investigate revenue growth opportunities"
            ]
        }
        
        if include_viz:
            report['visualizations'] = {
                'revenue_trend': self._generate_visualization('line', 'sales'),
                'customer_segments': 'Customer segmentation chart would be here'
            }
        
        return report
    
    def _generate_anomaly_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on detected anomalies."""
        if not anomalies:
            return ["No anomalies detected - data patterns are normal"]
        
        recommendations = []
        high_severity = [a for a in anomalies if a['severity'] == 'high']
        
        if high_severity:
            recommendations.append("Investigate high-severity anomalies immediately")
        
        if len(anomalies) > 10:
            recommendations.append("Multiple anomalies detected - review data collection process")
        
        return recommendations
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate model accuracy (mock implementation)."""
        # In a real implementation, this would use historical data to validate predictions
        return 0.85  # 85% accuracy
