# 🌍 Environmental Monitoring System
- AI-Powered Ecosystem Restoration & Multi-Agent Coordination Platform
- A comprehensive environmental monitoring system that combines IoT sensors, AI agents, predictive analytics, and community engagement to monitor ecosystem health and coordinate restoration efforts.

## 🚀 System Overview
- This system demonstrates a complete environmental monitoring solution with three main components:
1. **Backend System** (`environmental_monitoring.py`) - Core Python application
2. **Visualization Demo** (`index.html`) - Static HTML demonstration
3. **Interactive Dashboard** (web-based) - Full-featured user interface

## 📁 Project Structure
```
environmental-monitoring/
├── environmental_monitoring.py    # Core Python system & data models
├── web_dashboard.py              # Flask web application server
├── index.html                    # Static visualization demo
├── templates/                    # HTML templates for web dashboard
│   └── dashboard.html
├── environmental_system.db       # SQLite database (auto-created)
└── README.md                     # This file
```

## ⚡ Quick Start
### Option 1: Run the Full Web Dashboard (Recommended)
```bash
# Install dependencies
pip install flask flask-socketio numpy pandas sqlite3

# Run the web application
python web_dashboard.py

# Open browser to: http://localhost:5000
```

### Option 2: Run the Python System (Backend Only)
```bash
# Install dependencies
pip install numpy pandas sqlite3

# Run the core system
python environmental_monitoring.py
```

### Option 3: View HTML Demo (Static)
```bash
# Simply open in browser
open index.html
```

## 🎯 Key Features
### 🌐 Web Dashboard Features
- **Real-time monitoring** with live data updates via WebSocket
- **Interactive map visualization** with ecosystem locations
- **Health status indicators** with color-coded alerts
- **Agent status tracking** for drones and sensors
- **Community feedback integration** with reporting system
- **Restoration planning interface** with action management
- **Data export capabilities** for analysis and reporting

### 🔬 Environmental Monitoring
- **Real-time sensor data processing** (temperature, humidity, air quality, soil pH, water quality)
- **Multi-ecosystem support** (forests, wetlands, coastal areas, grasslands, deserts)
- **Health scoring algorithm** with weighted environmental factors
- **Trend prediction** using historical data analysis

### 🤖 Multi-Agent Coordination
- **Drone fleet management** for aerial monitoring
- **Ground sensor networks** for continuous data collection
- **Task assignment algorithms** for optimal resource allocation
- **Mission planning and tracking** for monitoring operations

### 📊 AI-Powered Analytics
- **Predictive modeling** for ecosystem health trends
- **Automated restoration recommendations** based on ecosystem state
- **Cost-benefit analysis** for intervention planning
- **Priority-based action scheduling**

### 👥 Community Integration
- **Citizen reporting system** for environmental issues
- **Community feedback processing** with verification
- **Crowdsourced data validation**
- **Public engagement tracking**

### 💾 Data Management
- **SQLite database** for persistent storage
- **RESTful data access patterns**
- **Historical data preservation**
- **Export capabilities** for analysis

## 🏗️ System Architecture
### Core Components
#### 1. Data Models
```python
- Location: Geographic coordinates with elevation
- SensorData: Environmental measurements from field devices
- EcosystemState: Complete ecosystem health profile
- RestorationAction: Planned interventions with timelines
- CommunityFeedback: Citizen reports and observations
```

#### 2. System Modules
```python
- DataStore: Database operations and data persistence
- EnvironmentalPredictor: AI analysis and health scoring
- MultiAgentCoordinator: Drone and sensor management
- CommunityEngagementModule: Public interaction handling
- EnvironmentalMonitoringSystem: Main orchestrator class
```

#### 3. Web Dashboard (`web_dashboard.py`)
```python
- Flask application server with REST API endpoints
- WebSocket integration for real-time updates
- Template rendering for dynamic web interface
- Bay Area ecosystem demonstration with realistic coordinates
- Live sensor data simulation and streaming
- Community feedback processing interface
```

#### 4. File Purposes
- **`environmental_monitoring.py`**: Core system engine with all business logic
- **`web_dashboard.py`**: Web application server that uses the core system
- **`index.html`**: Static demonstration of system output and UI concepts
