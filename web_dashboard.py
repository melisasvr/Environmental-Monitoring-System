from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import random

# Import your existing system
from environmental_monitoring import EnvironmentalMonitoringSystem, Location, SensorData, EcosystemType, HealthStatus

app = Flask(__name__)
app.config['SECRET_KEY'] = 'environmental_monitoring_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global system instance
monitoring_system = None

def init_system():
    """Initialize the monitoring system with demo data"""
    global monitoring_system
    
    # Import and setup your existing system
    from environmental_monitoring import create_demo_system
    monitoring_system = create_demo_system()
    
    # Add more diverse ecosystems for better map visualization - Bay Area locations
    additional_ecosystems = [
        {
            'id': 'coastal_001',
            'type': EcosystemType.COASTAL,
            'location': Location(37.7749, -122.4194, 5),  # San Francisco Bay
            'area': 18.3,
            'health_score': 72.0
        },
        {
            'id': 'grassland_001', 
            'type': EcosystemType.GRASSLAND,
            'location': Location(37.4419, -122.1430, 12),  # Palo Alto area
            'area': 31.7,
            'health_score': 58.0
        },
        {
            'id': 'forest_002',
            'type': EcosystemType.FOREST,
            'location': Location(37.3382, -122.0338, 25),  # San Jose hills
            'area': 42.1,
            'health_score': 43.0
        }
    ]
    
    # Add ecosystems to system
    from environmental_monitoring import EcosystemState
    for eco_data in additional_ecosystems:
        ecosystem = EcosystemState(
            ecosystem_id=eco_data['id'],
            ecosystem_type=eco_data['type'],
            location=eco_data['location'],
            area_km2=eco_data['area'],
            health_status=get_health_status_from_score(eco_data['health_score']),
            health_score=eco_data['health_score'],
            last_updated=datetime.now()
        )
        monitoring_system.register_ecosystem(ecosystem)
    
    print("✅ Monitoring system initialized with sample data")

def get_health_status_from_score(score):
    """Convert health score to status enum"""
    if score >= 80:
        return HealthStatus.EXCELLENT
    elif score >= 60:
        return HealthStatus.GOOD
    elif score >= 40:
        return HealthStatus.MODERATE
    elif score >= 20:
        return HealthStatus.POOR
    else:
        return HealthStatus.CRITICAL

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/ecosystems')
def get_ecosystems():
    """Get all ecosystems data for map"""
    if not monitoring_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    ecosystems_data = []
    for eco_id, ecosystem in monitoring_system.ecosystems.items():
        ecosystems_data.append({
            'id': eco_id,
            'type': ecosystem.ecosystem_type.value,
            'location': {
                'lat': ecosystem.location.latitude,
                'lng': ecosystem.location.longitude,
                'elevation': ecosystem.location.elevation
            },
            'area_km2': ecosystem.area_km2,
            'health_score': ecosystem.health_score,
            'health_status': ecosystem.health_status.value,
            'last_updated': ecosystem.last_updated.isoformat(),
            'sensor_count': len(ecosystem.sensor_data)
        })
    
    return jsonify(ecosystems_data)

@app.route('/api/agents')
def get_agents():
    """Get all agents data for map"""
    if not monitoring_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    agents_data = []
    for agent_id, agent in monitoring_system.coordinator.agents.items():
        # Generate random locations for demo (Bay Area locations)
        lat = 37.4419 + random.uniform(-0.1, 0.1)  # Around Palo Alto
        lng = -122.1430 + random.uniform(-0.1, 0.1)
        
        agents_data.append({
            'id': agent_id,
            'type': agent['type'],
            'status': agent['status'],
            'location': {'lat': lat, 'lng': lng},
            'battery_level': agent['battery_level'],
            'capabilities': agent['capabilities'],
            'last_update': agent['last_update'].isoformat()
        })
    
    return jsonify(agents_data)

@app.route('/api/actions')
def get_restoration_actions():
    """Get restoration actions for map"""
    if not monitoring_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    actions_data = []
    for action in monitoring_system.restoration_actions:
        actions_data.append({
            'id': action.action_id,
            'type': action.action_type.value,
            'location': {
                'lat': action.location.latitude,
                'lng': action.location.longitude
            },
            'priority': action.priority,
            'estimated_cost': action.estimated_cost,
            'expected_impact': action.expected_impact,
            'timeline_days': action.timeline_days,
            'resources_needed': action.resources_needed,
            'status': action.status
        })
    
    return jsonify(actions_data)

@app.route('/api/community-feedback')
def get_community_feedback():
    """Get community feedback for map"""
    if not monitoring_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    feedback_data = []
    for feedback in monitoring_system.community_module.feedback_queue:
        feedback_data.append({
            'id': feedback.feedback_id,
            'location': {
                'lat': feedback.location.latitude,
                'lng': feedback.location.longitude
            },
            'type': feedback.feedback_type,
            'priority': feedback.priority,
            'description': feedback.description,
            'timestamp': feedback.timestamp.isoformat(),
            'verified': feedback.verified
        })
    
    return jsonify(feedback_data)

@app.route('/api/sensor-data')
def get_recent_sensor_data():
    """Get recent sensor data"""
    if not monitoring_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    sensor_data = monitoring_system.data_store.get_recent_sensor_data(hours=24)
    data_list = []
    
    for data in sensor_data[-50:]:  # Last 50 readings
        data_list.append({
            'sensor_id': data.sensor_id,
            'timestamp': data.timestamp.isoformat(),
            'location': {
                'lat': data.location.latitude,
                'lng': data.location.longitude
            },
            'temperature': data.temperature,
            'humidity': data.humidity,
            'air_quality': data.air_quality,
            'soil_ph': data.soil_ph,
            'water_quality': data.water_quality,
            'biodiversity_index': data.biodiversity_index,
            'vegetation_density': data.vegetation_density
        })
    
    return jsonify(data_list)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to Environmental Monitoring System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_update')
def handle_update_request():
    """Handle real-time update requests"""
    if monitoring_system:
        # Send latest ecosystem data
        ecosystems_data = []
        for eco_id, ecosystem in monitoring_system.ecosystems.items():
            ecosystems_data.append({
                'id': eco_id,
                'health_score': ecosystem.health_score,
                'health_status': ecosystem.health_status.value,
                'last_updated': ecosystem.last_updated.isoformat()
            })
        
        emit('ecosystem_update', {'ecosystems': ecosystems_data})

def simulate_real_time_data():
    """Simulate real-time sensor data updates"""
    while True:
        if monitoring_system:
            # Generate random sensor data - Bay Area locations
            sensor_locations = [
                Location(37.4419 + random.uniform(-0.05, 0.05), 
                        -122.1430 + random.uniform(-0.05, 0.05)),  # Around Palo Alto
                Location(37.7749 + random.uniform(-0.05, 0.05), 
                        -122.4194 + random.uniform(-0.05, 0.05))   # Around San Francisco
            ]
            
            for i, location in enumerate(sensor_locations):
                sensor_data = SensorData(
                    sensor_id=f"sensor_{i+1:03d}",
                    timestamp=datetime.now(),
                    location=location,
                    temperature=20 + random.uniform(-5, 10),
                    humidity=60 + random.uniform(-20, 30),
                    air_quality=50 + random.uniform(-20, 40),
                    soil_ph=6.5 + random.uniform(-1, 1),
                    water_quality=70 + random.uniform(-20, 20),
                    noise_level=45 + random.uniform(-15, 30),
                    biodiversity_index=random.uniform(0.3, 0.8),
                    vegetation_density=random.uniform(0.4, 0.9)
                )
                
                monitoring_system.process_sensor_data(sensor_data)
            
            # Emit real-time update to connected clients
            socketio.emit('sensor_update', {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(sensor_locations)
            })
        
        time.sleep(10)  # Update every 10 seconds

# HTML Template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Environmental Monitoring Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .container {
            display: flex;
            height: calc(100vh - 100px);
        }
        
        .map-container {
            flex: 1;
            position: relative;
        }
        
        #map {
            height: 100%;
            width: 100%;
        }
        
        .sidebar {
            width: 350px;
            background: rgba(0, 0, 0, 0.9);
            padding: 20px;
            overflow-y: auto;
            backdrop-filter: blur(10px);
        }
        
        .status-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(5px);
        }
        
        .status-panel h3 {
            margin-top: 0;
            color: #00ff88;
        }
        
        .ecosystem-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }
        
        .health-excellent { border-left-color: #00ff88; }
        .health-good { border-left-color: #88ff00; }
        .health-moderate { border-left-color: #ffaa00; }
        .health-poor { border-left-color: #ff6600; }
        .health-critical { border-left-color: #ff4444; }
        
        .live-data {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .controls {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .btn {
            background: linear-gradient(45deg, #00ff88, #00ccff);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
        }
        
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 0.9em;
            z-index: 1000;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .online { background: #00ff88; }
        .offline { background: #ff4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌍 Environmental Monitoring Dashboard</h1>
        <div class="controls">
            <button class="btn" onclick="refreshData()">🔄 Refresh Data</button>
            <button class="btn" onclick="toggleLayers()">🗺️ Toggle Layers</button>
            <button class="btn" onclick="exportData()">📊 Export Data</button>
        </div>
    </div>
    
    <div class="container">
        <div class="map-container">
            <div id="map"></div>
            <div class="legend">
                <h4>🗺️ Map Legend</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background: #00ff88;"></div>
                    <span>Excellent Health</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #88ff00;"></div>
                    <span>Good Health</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffaa00;"></div>
                    <span>Moderate Health</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff6600;"></div>
                    <span>Poor Health</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff4444;"></div>
                    <span>Critical Health</span>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="status-panel">
                <h3>📊 System Status</h3>
                <div>
                    <span class="status-indicator online"></span>
                    <span>Connected</span>
                </div>
                <div>Last Update: <span id="last-update">--</span></div>
            </div>
            
            <div class="status-panel">
                <h3>🏞️ Ecosystems</h3>
                <div id="ecosystems-list">Loading...</div>
            </div>
            
            <div class="status-panel">
                <h3>🤖 Active Agents</h3>
                <div id="agents-list">Loading...</div>
            </div>
            
            <div class="status-panel">
                <h3>📡 Live Data Stream</h3>
                <div id="live-data">Waiting for data...</div>
            </div>
            
            <div class="status-panel">
                <h3>🛠️ Restoration Actions</h3>
                <div id="actions-list">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize map centered on Bay Area
        const map = L.map('map').setView([37.4419, -122.1430], 11);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        // Layer groups
        const ecosystemLayer = L.layerGroup().addTo(map);
        const agentLayer = L.layerGroup().addTo(map);
        const actionLayer = L.layerGroup().addTo(map);
        const feedbackLayer = L.layerGroup().addTo(map);
        
        // Socket.IO connection
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        });
        
        socket.on('ecosystem_update', function(data) {
            console.log('Ecosystem update received:', data);
            updateEcosystemsList(data.ecosystems);
        });
        
        socket.on('sensor_update', function(data) {
            console.log('Sensor update received:', data);
            document.getElementById('live-data').innerHTML = 
                `<div class="live-data">📡 ${data.timestamp}: ${data.data_points} sensors updated</div>` +
                document.getElementById('live-data').innerHTML;
        });
        
        // Health status colors
        const healthColors = {
            'excellent': '#00ff88',
            'good': '#88ff00',
            'moderate': '#ffaa00',
            'poor': '#ff6600',
            'critical': '#ff4444'
        };
        
        // Load and display ecosystems
        function loadEcosystems() {
            fetch('/api/ecosystems')
                .then(response => response.json())
                .then(data => {
                    ecosystemLayer.clearLayers();
                    updateEcosystemsList(data);
                    
                    data.forEach(ecosystem => {
                        const color = healthColors[ecosystem.health_status];
                        const marker = L.circleMarker([ecosystem.location.lat, ecosystem.location.lng], {
                            radius: 8 + (ecosystem.health_score / 10),
                            fillColor: color,
                            color: '#fff',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.8
                        }).addTo(ecosystemLayer);
                        
                        marker.bindPopup(`
                            <div style="color: #000;">
                                <h3>${ecosystem.id}</h3>
                                <p><strong>Type:</strong> ${ecosystem.type}</p>
                                <p><strong>Health:</strong> ${ecosystem.health_status} (${ecosystem.health_score.toFixed(1)})</p>
                                <p><strong>Area:</strong> ${ecosystem.area_km2} km²</p>
                                <p><strong>Sensors:</strong> ${ecosystem.sensor_count}</p>
                                <p><strong>Updated:</strong> ${new Date(ecosystem.last_updated).toLocaleString()}</p>
                            </div>
                        `);
                    });
                });
        }
        
        // Load and display agents
        function loadAgents() {
            fetch('/api/agents')
                .then(response => response.json())
                .then(data => {
                    agentLayer.clearLayers();
                    updateAgentsList(data);
                    
                    data.forEach(agent => {
                        const color = agent.status === 'idle' ? '#00ff88' : '#ffaa00';
                        const icon = agent.type === 'monitoring_drone' ? '🚁' : '📡';
                        
                        const marker = L.marker([agent.location.lat, agent.location.lng], {
                            icon: L.divIcon({
                                html: `<div style="background: ${color}; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 16px;">${icon}</div>`,
                                iconSize: [30, 30],
                                className: 'agent-marker'
                            })
                        }).addTo(agentLayer);
                        
                        marker.bindPopup(`
                            <div style="color: #000;">
                                <h3>${agent.id}</h3>
                                <p><strong>Type:</strong> ${agent.type}</p>
                                <p><strong>Status:</strong> ${agent.status}</p>
                                <p><strong>Battery:</strong> ${agent.battery_level}%</p>
                                <p><strong>Capabilities:</strong> ${agent.capabilities.join(', ')}</p>
                            </div>
                        `);
                    });
                });
        }
        
        // Load and display restoration actions
        function loadActions() {
            fetch('/api/actions')
                .then(response => response.json())
                .then(data => {
                    actionLayer.clearLayers();
                    updateActionsList(data);
                    
                    data.forEach(action => {
                        const color = action.priority === 1 ? '#ff4444' : action.priority <= 3 ? '#ffaa00' : '#00ff88';
                        
                        const marker = L.marker([action.location.lat, action.location.lng], {
                            icon: L.divIcon({
                                html: `<div style="background: ${color}; border-radius: 50%; width: 25px; height: 25px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold;">${action.priority}</div>`,
                                iconSize: [25, 25],
                                className: 'action-marker'
                            })
                        }).addTo(actionLayer);
                        
                        marker.bindPopup(`
                            <div style="color: #000;">
                                <h3>${action.type.replace('_', ' ').toUpperCase()}</h3>
                                <p><strong>Priority:</strong> ${action.priority}</p>
                                <p><strong>Cost:</strong> $${action.estimated_cost.toLocaleString()}</p>
                                <p><strong>Impact:</strong> ${(action.expected_impact * 100).toFixed(1)}%</p>
                                <p><strong>Timeline:</strong> ${action.timeline_days} days</p>
                                <p><strong>Status:</strong> ${action.status}</p>
                            </div>
                        `);
                    });
                });
        }
        
        // Update sidebar lists
        function updateEcosystemsList(ecosystems) {
            const container = document.getElementById('ecosystems-list');
            container.innerHTML = ecosystems.map(eco => `
                <div class="ecosystem-item health-${eco.health_status}">
                    <div><strong>${eco.id}</strong></div>
                    <div>${eco.type} - ${eco.health_score.toFixed(1)}%</div>
                </div>
            `).join('');
        }
        
        function updateAgentsList(agents) {
            const container = document.getElementById('agents-list');
            container.innerHTML = agents.map(agent => `
                <div class="ecosystem-item">
                    <div><strong>${agent.id}</strong></div>
                    <div>${agent.type} - ${agent.status}</div>
                    <div>Battery: ${agent.battery_level}%</div>
                </div>
            `).join('');
        }
        
        function updateActionsList(actions) {
            const container = document.getElementById('actions-list');
            container.innerHTML = actions.slice(0, 5).map(action => `
                <div class="ecosystem-item">
                    <div><strong>Priority ${action.priority}</strong></div>
                    <div>${action.type.replace('_', ' ')}</div>
                    <div>Impact: ${(action.expected_impact * 100).toFixed(1)}%</div>
                </div>
            `).join('');
        }
        
        // Control functions
        function refreshData() {
            loadEcosystems();
            loadAgents();
            loadActions();
            socket.emit('request_update');
        }
        
        function toggleLayers() {
            if (map.hasLayer(agentLayer)) {
                map.removeLayer(agentLayer);
                map.removeLayer(actionLayer);
            } else {
                map.addLayer(agentLayer);
                map.addLayer(actionLayer);
            }
        }
        
        function exportData() {
            window.open('/api/ecosystems', '_blank');
        }
        
        // Initialize dashboard
        loadEcosystems();
        loadAgents();
        loadActions();
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
import os
os.makedirs('templates', exist_ok=True)

# FIX: Specify UTF-8 encoding when writing the file
with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(DASHBOARD_HTML)

if __name__ == '__main__':
    print("🚀 Starting Environmental Monitoring Dashboard...")
    
    # Initialize the monitoring system
    init_system()
    
    # Start real-time data simulation in background
    data_thread = threading.Thread(target=simulate_real_time_data, daemon=True)
    data_thread.start()
    
    print("✅ Dashboard ready!")
    print("🌐 Open your browser to: http://localhost:5000")
    print("🗺️ Interactive map with real-time ecosystem monitoring")
    print("📊 Live data updates every 10 seconds")
    print("🔄 Auto-refresh every 30 seconds")
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)