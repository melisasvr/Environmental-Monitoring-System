import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod
import json
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class EcosystemType(Enum):
    FOREST = "forest"
    WETLAND = "wetland"
    COASTAL = "coastal"
    GRASSLAND = "grassland"
    DESERT = "desert"

class HealthStatus(Enum):
    CRITICAL = "critical"
    POOR = "poor"
    MODERATE = "moderate"
    GOOD = "good"
    EXCELLENT = "excellent"

class ActionType(Enum):
    REFORESTATION = "reforestation"
    POLLUTION_CLEANUP = "pollution_cleanup"
    SOIL_RESTORATION = "soil_restoration"
    WATER_TREATMENT = "water_treatment"
    HABITAT_CREATION = "habitat_creation"

@dataclass
class Location:
    latitude: float
    longitude: float
    elevation: float = 0.0
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance between two locations (simplified)"""
        return np.sqrt((self.latitude - other.latitude)**2 + 
                      (self.longitude - other.longitude)**2)

@dataclass
class SensorData:
    sensor_id: str
    timestamp: datetime
    location: Location
    temperature: float
    humidity: float
    air_quality: float
    soil_ph: float
    water_quality: float
    noise_level: float
    biodiversity_index: float
    vegetation_density: float

@dataclass
class EcosystemState:
    ecosystem_id: str
    ecosystem_type: EcosystemType
    location: Location
    area_km2: float
    health_status: HealthStatus
    health_score: float
    last_updated: datetime
    sensor_data: List[SensorData] = field(default_factory=list)
    predicted_trends: Dict[str, float] = field(default_factory=dict)

@dataclass
class RestorationAction:
    action_id: str
    action_type: ActionType
    location: Location
    priority: int
    estimated_cost: float
    expected_impact: float
    timeline_days: int
    resources_needed: List[str]
    status: str = "planned"

@dataclass
class CommunityFeedback:
    feedback_id: str
    location: Location
    timestamp: datetime
    feedback_type: str
    priority: int
    description: str
    contact_info: str
    verified: bool = False

# Core System Components
class DataStore:
    """Centralized data storage and management"""
    
    def __init__(self, db_path: str = "environmental_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY,
                sensor_id TEXT,
                timestamp TEXT,
                location_lat REAL,
                location_lon REAL,
                temperature REAL,
                humidity REAL,
                air_quality REAL,
                soil_ph REAL,
                water_quality REAL,
                noise_level REAL,
                biodiversity_index REAL,
                vegetation_density REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ecosystem_states (
                id INTEGER PRIMARY KEY,
                ecosystem_id TEXT,
                ecosystem_type TEXT,
                location_lat REAL,
                location_lon REAL,
                area_km2 REAL,
                health_status TEXT,
                health_score REAL,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restoration_actions (
                id INTEGER PRIMARY KEY,
                action_id TEXT,
                action_type TEXT,
                location_lat REAL,
                location_lon REAL,
                priority INTEGER,
                estimated_cost REAL,
                expected_impact REAL,
                timeline_days INTEGER,
                resources_needed TEXT,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS community_feedback (
                id INTEGER PRIMARY KEY,
                feedback_id TEXT,
                location_lat REAL,
                location_lon REAL,
                timestamp TEXT,
                feedback_type TEXT,
                priority INTEGER,
                description TEXT,
                contact_info TEXT,
                verified BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_sensor_data(self, data: SensorData):
        """Store sensor data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data 
            (sensor_id, timestamp, location_lat, location_lon, temperature, 
             humidity, air_quality, soil_ph, water_quality, noise_level, 
             biodiversity_index, vegetation_density)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.sensor_id, data.timestamp.isoformat(), data.location.latitude,
            data.location.longitude, data.temperature, data.humidity,
            data.air_quality, data.soil_ph, data.water_quality,
            data.noise_level, data.biodiversity_index, data.vegetation_density
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_sensor_data(self, hours: int = 24) -> List[SensorData]:
        """Retrieve recent sensor data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute('''
            SELECT * FROM sensor_data 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        sensor_data = []
        for row in rows:
            data = SensorData(
                sensor_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                location=Location(row[3], row[4]),
                temperature=row[5],
                humidity=row[6],
                air_quality=row[7],
                soil_ph=row[8],
                water_quality=row[9],
                noise_level=row[10],
                biodiversity_index=row[11],
                vegetation_density=row[12]
            )
            sensor_data.append(data)
        
        return sensor_data

class EnvironmentalPredictor:
    """AI-powered environmental prediction system"""
    
    def __init__(self):
        self.models = {}
        self.feature_weights = {
            'temperature': 0.15,
            'humidity': 0.12,
            'air_quality': 0.18,
            'soil_ph': 0.14,
            'water_quality': 0.16,
            'noise_level': 0.05,
            'biodiversity_index': 0.12,
            'vegetation_density': 0.08
        }
    
    def calculate_health_score(self, sensor_data: List[SensorData]) -> float:
        """Calculate ecosystem health score from sensor data"""
        if not sensor_data:
            return 0.0
        
        # Average recent readings
        recent_data = sensor_data[-10:]  # Last 10 readings
        avg_values = {}
        
        for key in self.feature_weights.keys():
            values = [getattr(data, key) for data in recent_data]
            avg_values[key] = np.mean(values)
        
        # Normalize and weight values (simplified scoring)
        health_score = 0.0
        for feature, weight in self.feature_weights.items():
            # Normalize to 0-1 scale (simplified)
            normalized = self._normalize_feature(feature, avg_values[feature])
            health_score += normalized * weight
        
        return min(100.0, max(0.0, health_score * 100))
    
    def _normalize_feature(self, feature: str, value: float) -> float:
        """Normalize feature values to 0-1 scale"""
        # Simplified normalization - in practice, use domain-specific ranges
        normalization_ranges = {
            'temperature': (0, 40),
            'humidity': (0, 100),
            'air_quality': (0, 500),  # AQI scale
            'soil_ph': (0, 14),
            'water_quality': (0, 100),
            'noise_level': (0, 120),
            'biodiversity_index': (0, 1),
            'vegetation_density': (0, 1)
        }
        
        min_val, max_val = normalization_ranges.get(feature, (0, 100))
        
        # For features where higher is better
        if feature in ['humidity', 'water_quality', 'biodiversity_index', 'vegetation_density']:
            return (value - min_val) / (max_val - min_val)
        # For soil_ph, optimal is around 6.5-7.5
        elif feature == 'soil_ph':
            optimal = 7.0
            return 1.0 - abs(value - optimal) / 7.0
        # For features where lower is better
        else:
            return 1.0 - (value - min_val) / (max_val - min_val)
    
    def predict_ecosystem_trends(self, ecosystem: EcosystemState) -> Dict[str, float]:
        """Predict ecosystem health trends"""
        if not ecosystem.sensor_data:
            return {}
        
        # Simple trend analysis (in practice, use machine learning)
        trends = {}
        
        # Analyze last 30 days of data
        recent_data = [d for d in ecosystem.sensor_data 
                      if d.timestamp > datetime.now() - timedelta(days=30)]
        
        if len(recent_data) >= 2:
            # Calculate trend slope for each metric
            for metric in ['temperature', 'air_quality', 'biodiversity_index']:
                values = [getattr(d, metric) for d in recent_data]
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    trends[f"{metric}_trend"] = slope
        
        return trends
    
    def recommend_actions(self, ecosystem: EcosystemState) -> List[RestorationAction]:
        """Recommend restoration actions based on ecosystem state"""
        actions = []
        
        if ecosystem.health_score < 40:
            # Critical state - immediate action needed
            if ecosystem.ecosystem_type == EcosystemType.FOREST:
                actions.append(RestorationAction(
                    action_id=f"reforest_{ecosystem.ecosystem_id}_{datetime.now().strftime('%Y%m%d')}",
                    action_type=ActionType.REFORESTATION,
                    location=ecosystem.location,
                    priority=1,
                    estimated_cost=50000 * ecosystem.area_km2,
                    expected_impact=0.7,
                    timeline_days=180,
                    resources_needed=["tree_saplings", "planting_equipment", "water_supply"]
                ))
            
            elif ecosystem.ecosystem_type == EcosystemType.WETLAND:
                actions.append(RestorationAction(
                    action_id=f"water_treat_{ecosystem.ecosystem_id}_{datetime.now().strftime('%Y%m%d')}",
                    action_type=ActionType.WATER_TREATMENT,
                    location=ecosystem.location,
                    priority=1,
                    estimated_cost=30000 * ecosystem.area_km2,
                    expected_impact=0.6,
                    timeline_days=90,
                    resources_needed=["filtration_systems", "aquatic_plants", "monitoring_equipment"]
                ))
        
        elif ecosystem.health_score < 70:
            # Moderate intervention needed
            actions.append(RestorationAction(
                action_id=f"soil_restore_{ecosystem.ecosystem_id}_{datetime.now().strftime('%Y%m%d')}",
                action_type=ActionType.SOIL_RESTORATION,
                location=ecosystem.location,
                priority=2,
                estimated_cost=20000 * ecosystem.area_km2,
                expected_impact=0.4,
                timeline_days=60,
                resources_needed=["soil_amendments", "native_seeds", "mulch"]
            ))
        
        return actions

class MultiAgentCoordinator:
    """Coordinates multiple AI agents for environmental monitoring"""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.active_missions = {}
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Register a new agent (drone, sensor, etc.)"""
        self.agents[agent_id] = {
            'type': agent_type,
            'capabilities': capabilities,
            'status': 'idle',
            'location': None,
            'battery_level': 100,
            'last_update': datetime.now()
        }
    
    def assign_monitoring_task(self, ecosystem: EcosystemState, task_type: str) -> str:
        """Assign monitoring task to available agents"""
        available_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent['status'] == 'idle' and task_type in agent['capabilities']
        ]
        
        if not available_agents:
            return None
        
        # Select best agent based on proximity and capabilities
        selected_agent = available_agents[0]  # Simplified selection
        
        mission_id = f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_missions[mission_id] = {
            'agent_id': selected_agent,
            'ecosystem_id': ecosystem.ecosystem_id,
            'task_type': task_type,
            'start_time': datetime.now(),
            'status': 'active'
        }
        
        self.agents[selected_agent]['status'] = 'busy'
        
        logger.info(f"Assigned {task_type} task to agent {selected_agent} for ecosystem {ecosystem.ecosystem_id}")
        return mission_id
    
    def update_agent_status(self, agent_id: str, status: str, location: Location = None):
        """Update agent status and location"""
        if agent_id in self.agents:
            self.agents[agent_id]['status'] = status
            self.agents[agent_id]['last_update'] = datetime.now()
            if location:
                self.agents[agent_id]['location'] = location

class CommunityEngagementModule:
    """Handles community feedback and engagement"""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.feedback_queue = []
    
    def submit_feedback(self, feedback: CommunityFeedback):
        """Submit community feedback"""
        self.feedback_queue.append(feedback)
        
        # Store in database
        conn = sqlite3.connect(self.data_store.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO community_feedback 
            (feedback_id, location_lat, location_lon, timestamp, feedback_type, 
             priority, description, contact_info, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id, feedback.location.latitude, feedback.location.longitude,
            feedback.timestamp.isoformat(), feedback.feedback_type, feedback.priority,
            feedback.description, feedback.contact_info, feedback.verified
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Community feedback submitted: {feedback.feedback_id}")
    
    def process_feedback(self) -> List[RestorationAction]:
        """Process community feedback and generate actions"""
        actions = []
        
        for feedback in self.feedback_queue:
            if feedback.priority >= 7:  # High priority feedback
                action = RestorationAction(
                    action_id=f"community_{feedback.feedback_id}",
                    action_type=ActionType.POLLUTION_CLEANUP,  # Default action
                    location=feedback.location,
                    priority=feedback.priority,
                    estimated_cost=5000,
                    expected_impact=0.3,
                    timeline_days=30,
                    resources_needed=["cleanup_crew", "equipment"]
                )
                actions.append(action)
        
        # Clear processed feedback
        self.feedback_queue = []
        
        return actions

class EnvironmentalMonitoringSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.data_store = DataStore()
        self.predictor = EnvironmentalPredictor()
        self.coordinator = MultiAgentCoordinator()
        self.community_module = CommunityEngagementModule(self.data_store)
        self.ecosystems = {}
        self.restoration_actions = []
    
    def register_ecosystem(self, ecosystem: EcosystemState):
        """Register a new ecosystem for monitoring"""
        self.ecosystems[ecosystem.ecosystem_id] = ecosystem
        logger.info(f"Registered ecosystem: {ecosystem.ecosystem_id}")
    
    def process_sensor_data(self, sensor_data: SensorData):
        """Process incoming sensor data"""
        # Store data
        self.data_store.store_sensor_data(sensor_data)
        
        # Find relevant ecosystem
        for ecosystem in self.ecosystems.values():
            if ecosystem.location.distance_to(sensor_data.location) < 0.1:  # Within 0.1 units
                ecosystem.sensor_data.append(sensor_data)
                
                # Update health score
                ecosystem.health_score = self.predictor.calculate_health_score(ecosystem.sensor_data)
                ecosystem.health_status = self._get_health_status(ecosystem.health_score)
                ecosystem.last_updated = datetime.now()
                
                # Predict trends
                ecosystem.predicted_trends = self.predictor.predict_ecosystem_trends(ecosystem)
                
                break
    
    def _get_health_status(self, health_score: float) -> HealthStatus:
        """Convert health score to status"""
        if health_score >= 80:
            return HealthStatus.EXCELLENT
        elif health_score >= 60:
            return HealthStatus.GOOD
        elif health_score >= 40:
            return HealthStatus.MODERATE
        elif health_score >= 20:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def generate_restoration_plan(self) -> List[RestorationAction]:
        """Generate comprehensive restoration plan"""
        all_actions = []
        
        # Generate actions based on ecosystem health
        for ecosystem in self.ecosystems.values():
            actions = self.predictor.recommend_actions(ecosystem)
            all_actions.extend(actions)
        
        # Add community-driven actions
        community_actions = self.community_module.process_feedback()
        all_actions.extend(community_actions)
        
        # Sort by priority and expected impact
        all_actions.sort(key=lambda x: (x.priority, -x.expected_impact))
        
        return all_actions
    
    def execute_monitoring_cycle(self):
        """Execute one complete monitoring cycle"""
        logger.info("Starting monitoring cycle...")
        
        # Assign monitoring tasks to agents
        for ecosystem in self.ecosystems.values():
            if ecosystem.health_status in [HealthStatus.CRITICAL, HealthStatus.POOR]:
                mission_id = self.coordinator.assign_monitoring_task(ecosystem, "environmental_monitoring")
                if mission_id:
                    logger.info(f"Assigned monitoring mission {mission_id} to ecosystem {ecosystem.ecosystem_id}")
        
        # Generate restoration plan
        restoration_actions = self.generate_restoration_plan()
        self.restoration_actions = restoration_actions
        
        logger.info(f"Generated {len(restoration_actions)} restoration actions")
        
        # Print summary
        self.print_system_status()
    
    def print_system_status(self):
        """Print current system status"""
        print("\n" + "="*60)
        print("ENVIRONMENTAL MONITORING SYSTEM STATUS")
        print("="*60)
        
        print(f"\nRegistered Ecosystems: {len(self.ecosystems)}")
        for eco_id, ecosystem in self.ecosystems.items():
            print(f"  {eco_id}: {ecosystem.health_status.value} (Score: {ecosystem.health_score:.1f})")
        
        print(f"\nActive Agents: {len(self.coordinator.agents)}")
        for agent_id, agent in self.coordinator.agents.items():
            print(f"  {agent_id}: {agent['type']} - {agent['status']}")
        
        print(f"\nPending Restoration Actions: {len(self.restoration_actions)}")
        for action in self.restoration_actions[:5]:  # Show top 5
            print(f"  {action.action_type.value} - Priority: {action.priority}, Impact: {action.expected_impact}")
        
        print("\n" + "="*60)

# Demo Usage
def create_demo_system():
    """Create a demonstration system with sample data"""
    system = EnvironmentalMonitoringSystem()
    
    # Register some ecosystems
    forest_ecosystem = EcosystemState(
        ecosystem_id="forest_001",
        ecosystem_type=EcosystemType.FOREST,
        location=Location(40.7128, -74.0060),
        area_km2=25.5,
        health_status=HealthStatus.MODERATE,
        health_score=65.0,
        last_updated=datetime.now()
    )
    
    wetland_ecosystem = EcosystemState(
        ecosystem_id="wetland_001",
        ecosystem_type=EcosystemType.WETLAND,
        location=Location(40.7580, -73.9855),
        area_km2=15.2,
        health_status=HealthStatus.POOR,
        health_score=35.0,
        last_updated=datetime.now()
    )
    
    system.register_ecosystem(forest_ecosystem)
    system.register_ecosystem(wetland_ecosystem)
    
    # Register some agents
    system.coordinator.register_agent("drone_001", "monitoring_drone", ["environmental_monitoring", "aerial_survey"])
    system.coordinator.register_agent("sensor_001", "ground_sensor", ["environmental_monitoring", "soil_analysis"])
    
    # Generate some sample sensor data
    sample_data = [
        SensorData(
            sensor_id="sensor_001",
            timestamp=datetime.now(),
            location=Location(40.7128, -74.0060),
            temperature=22.5,
            humidity=65.0,
            air_quality=45.0,
            soil_ph=6.8,
            water_quality=78.0,
            noise_level=55.0,
            biodiversity_index=0.6,
            vegetation_density=0.7
        ),
        SensorData(
            sensor_id="sensor_002",
            timestamp=datetime.now(),
            location=Location(40.7580, -73.9855),
            temperature=24.1,
            humidity=80.0,
            air_quality=85.0,
            soil_ph=5.5,
            water_quality=45.0,
            noise_level=70.0,
            biodiversity_index=0.3,
            vegetation_density=0.4
        )
    ]
    
    # Process sensor data
    for data in sample_data:
        system.process_sensor_data(data)
    
    # Add community feedback
    feedback = CommunityFeedback(
        feedback_id="feedback_001",
        location=Location(40.7128, -74.0060),
        timestamp=datetime.now(),
        feedback_type="pollution_report",
        priority=8,
        description="Unusual smell and discoloration in the water near the forest area",
        contact_info="citizen@email.com"
    )
    
    system.community_module.submit_feedback(feedback)
    
    return system

if __name__ == "__main__":
    # Create and run demo system
    demo_system = create_demo_system()
    demo_system.execute_monitoring_cycle()
    
    print("\nDemo system created successfully!")
    print("You can now extend this system with:")
    print("- Computer vision modules for image analysis")
    print("- Machine learning models for better predictions")
    print("- Real drone/sensor integration")
    print("- Advanced multi-agent coordination")
    print("- Web dashboard for visualization")
    print("- Mobile app for community engagement")