/**
 * Saarthi AI — 3D LiDAR Perception & Autonomous Decision Engine
 * ==============================================================================
 * Interactive WebGL 3D Point Cloud Viewer (Three.js) + 2D Tactical BEV Canvas
 * Autonomous Decision Banner & Risk Assessment based on LiDAR Readings
 * ==============================================================================
 */

(function () {
  'use strict';

  // Global LiDAR Demo State
  const state = {
    initialized: false,
    currentSampleId: 'sample_night_rain',
    scoreThreshold: 0.40,
    activeFilterClass: 'ALL',
    activeViewMode: 'isometric', // 'isometric', 'top', 'driver'
    showPoints: true,
    showBoxes: true,
    analysisData: null,
    hoveredObjIndex: null,
    availableSamples: []
  };

  // Three.js Scene References
  let scene, camera, renderer, controls;
  let pointCloudMesh = null;
  let boxesGroup = null;
  let busGroup = null;
  let gridHelper = null;
  let animFrameId = null;

  // BEV Canvas References
  let bevCanvas = null;
  let bevCtx = null;

  // DOM Elements Cache
  const dom = {};

  // Wait for DOM
  document.addEventListener('DOMContentLoaded', initLidarDemo);

  /**
   * Initializes the LiDAR Demo Module
   */
  function initLidarDemo() {
    cacheDomElements();
    setupModalitySwitcher();
    setupEventListeners();
    fetchLidarStatus();
    fetchLidarSamples();
  }

  /**
   * Caches all required DOM elements
   */
  function cacheDomElements() {
    // Modality bar
    dom.tabCamera = document.getElementById('tab-modality-camera');
    dom.tabLidar = document.getElementById('tab-modality-lidar');
    dom.cameraContainer = document.getElementById('camera-perception-container');
    dom.lidarContainer = document.getElementById('lidar-perception-container');

    // LiDAR UI elements
    dom.presetsList = document.getElementById('lidar-presets-list');
    dom.dropzone = document.getElementById('lidar-dropzone');
    dom.fileInput = document.getElementById('lidar-file-input');
    dom.loadedFilename = document.getElementById('lidar-loaded-filename');
    dom.loadedPointsCount = document.getElementById('lidar-loaded-pts');
    dom.loadedCard = document.getElementById('lidar-loaded-card');

    // Controls
    dom.confSlider = document.getElementById('lidar-conf-slider');
    dom.confValue = document.getElementById('lidar-conf-val');
    dom.filterPills = document.querySelectorAll('.lidar-filter-pill');
    dom.btnRunLidar = document.getElementById('btn-run-lidar');

    // Decision Banner
    dom.decisionBadge = document.getElementById('lidar-decision-badge');
    dom.decisionRiskScore = document.getElementById('lidar-risk-score');
    dom.decisionRiskLevel = document.getElementById('lidar-risk-level');
    dom.decisionSpeed = document.getElementById('lidar-safe-speed');
    dom.decisionClosestDist = document.getElementById('lidar-closest-dist');
    dom.decisionReason = document.getElementById('lidar-decision-reason');
    dom.zoneFront = document.getElementById('lidar-zone-front');
    dom.zoneLeft = document.getElementById('lidar-zone-left');
    dom.zoneRight = document.getElementById('lidar-zone-right');

    // Viewports
    dom.threeViewport = document.getElementById('lidar-three-container');
    dom.bevCanvas = document.getElementById('lidar-bev-canvas');
    dom.hudPoints = document.getElementById('lidar-hud-points');
    dom.hudLatency = document.getElementById('lidar-hud-latency');
    dom.hudBoxes = document.getElementById('lidar-hud-boxes');

    // Stats
    dom.statTotalDetections = document.getElementById('lidar-stat-total-det');
    dom.statCars = document.getElementById('lidar-stat-cars');
    dom.statPedestrians = document.getElementById('lidar-stat-pedestrians');
    dom.statLatency = document.getElementById('lidar-stat-latency');
    dom.detectionsList = document.getElementById('lidar-detections-list');

    // Viewport Tools
    dom.btnViewTop = document.getElementById('btn-lidar-view-top');
    dom.btnViewDriver = document.getElementById('btn-lidar-view-driver');
    dom.btnViewIso = document.getElementById('btn-lidar-view-iso');
    dom.btnTogglePoints = document.getElementById('btn-lidar-toggle-pts');
    dom.btnToggleBoxes = document.getElementById('btn-lidar-toggle-boxes');
    dom.btnResetCam = document.getElementById('btn-lidar-reset-cam');
  }

  /**
   * Sets up seamless navigation between Camera Perception and LiDAR Perception
   */
  function setupModalitySwitcher() {
    if (!dom.tabCamera || !dom.tabLidar) return;

    dom.tabCamera.addEventListener('click', () => {
      switchModality('camera');
    });

    dom.tabLidar.addEventListener('click', () => {
      switchModality('lidar');
    });
  }

  function switchModality(mode) {
    if (mode === 'camera') {
      dom.tabCamera.classList.add('active');
      dom.tabLidar.classList.remove('active');
      if (dom.cameraContainer) dom.cameraContainer.style.display = 'block';
      if (dom.lidarContainer) dom.lidarContainer.style.display = 'none';
    } else if (mode === 'lidar') {
      dom.tabLidar.classList.add('active');
      dom.tabCamera.classList.remove('active');
      if (dom.cameraContainer) dom.cameraContainer.style.display = 'none';
      if (dom.lidarContainer) dom.lidarContainer.style.display = 'block';

      // Lazy-init 3D & 2D renderers upon first reveal
      if (!state.initialized) {
        initThreeJs();
        initBevCanvas();
        state.initialized = true;

        // Auto-run first preset if not loaded yet
        runLidarAnalysis({ sample_id: state.currentSampleId });
      } else {
        // Trigger resize to adapt to visible container dimensions
        onWindowResize();
      }
    }
  }

  /**
   * Sets up UI event listeners
   */
  function setupEventListeners() {
    // Confidence threshold slider
    if (dom.confSlider) {
      dom.confSlider.addEventListener('input', (e) => {
        state.scoreThreshold = parseFloat(e.target.value);
        if (dom.confValue) dom.confValue.textContent = state.scoreThreshold.toFixed(2);
      });
      dom.confSlider.addEventListener('change', () => {
        // Re-analyze or re-filter
        runLidarAnalysis({ sample_id: state.currentSampleId });
      });
    }

    // Filter pills (ALL / CAR / PEDESTRIAN)
    if (dom.filterPills) {
      dom.filterPills.forEach((pill) => {
        pill.addEventListener('click', () => {
          dom.filterPills.forEach((p) => p.classList.remove('active'));
          pill.classList.add('active');
          state.activeFilterClass = pill.dataset.filter || 'ALL';
          renderFilteredViews();
        });
      });
    }

    // Run Button
    if (dom.btnRunLidar) {
      dom.btnRunLidar.addEventListener('click', () => {
        runLidarAnalysis({ sample_id: state.currentSampleId });
      });
    }

    // Drag and Drop Upload
    if (dom.dropzone && dom.fileInput) {
      dom.dropzone.addEventListener('click', () => dom.fileInput.click());

      dom.dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.dropzone.classList.add('dragover');
      });

      dom.dropzone.addEventListener('dragleave', () => {
        dom.dropzone.classList.remove('dragover');
      });

      dom.dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          uploadAndAnalyzeFile(e.dataTransfer.files[0]);
        }
      });

      dom.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
          uploadAndAnalyzeFile(e.target.files[0]);
        }
      });
    }

    // Viewport Controls
    if (dom.btnViewTop) dom.btnViewTop.addEventListener('click', () => setCameraView('top'));
    if (dom.btnViewDriver) dom.btnViewDriver.addEventListener('click', () => setCameraView('driver'));
    if (dom.btnViewIso) dom.btnViewIso.addEventListener('click', () => setCameraView('isometric'));
    if (dom.btnResetCam) dom.btnResetCam.addEventListener('click', () => setCameraView('isometric'));

    if (dom.btnTogglePoints) {
      dom.btnTogglePoints.addEventListener('click', () => {
        state.showPoints = !state.showPoints;
        if (pointCloudMesh) pointCloudMesh.visible = state.showPoints;
        dom.btnTogglePoints.classList.toggle('active', state.showPoints);
        drawBevView();
      });
    }

    if (dom.btnToggleBoxes) {
      dom.btnToggleBoxes.addEventListener('click', () => {
        state.showBoxes = !state.showBoxes;
        if (boxesGroup) boxesGroup.visible = state.showBoxes;
        dom.btnToggleBoxes.classList.toggle('active', state.showBoxes);
        drawBevView();
      });
    }

    window.addEventListener('resize', onWindowResize);
  }

  /**
   * Fetches LiDAR Model Status from Backend API
   */
  async function fetchLidarStatus() {
    try {
      const res = await fetch('/api/lidar/status');
      if (!res.ok) return;
      const data = await res.json();
      const statusPill = document.getElementById('lidar-status-pill');
      if (statusPill && data.status === 'READY') {
        statusPill.innerHTML = '<span class="status-dot green"></span> LiDAR MODEL READY';
      }
      const modelTag = document.getElementById('lidar-model-name-tag');
      if (modelTag && data.model_name) {
        modelTag.textContent = data.model_name + ' (CPU)';
      }
    } catch (e) {
      console.warn('[LiDAR] Failed to fetch status:', e);
    }
  }

  /**
   * Fetches Real Curated LiDAR Point Cloud Samples
   */
  async function fetchLidarSamples() {
    try {
      const res = await fetch('/api/lidar/samples');
      if (!res.ok) return;
      const data = await res.json();
      state.availableSamples = data.samples || [];
      renderPresetsList();
    } catch (e) {
      console.warn('[LiDAR] Failed to fetch samples:', e);
    }
  }

  /**
   * Renders the Curated Preset Buttons
   */
  function renderPresetsList() {
    if (!dom.presetsList || !state.availableSamples.length) return;
    dom.presetsList.innerHTML = '';

    const iconMap = {
      sample_night_rain: '🌧️',
      sample_urban_crossing: '🚶',
      sample_multi_vehicle: '🚗',
      sample_bus_stop: '🚏',
      sample_night_highway: '🛣️'
    };

    state.availableSamples.forEach((sample, idx) => {
      const btn = document.createElement('button');
      btn.className = `lidar-preset-btn ${sample.id === state.currentSampleId ? 'active' : ''}`;
      btn.dataset.sampleId = sample.id;

      const icon = iconMap[sample.id] || '📡';
      btn.innerHTML = `
        <div class="preset-btn-left">
          <div class="preset-info-name">${icon} ${sample.name}</div>
          <div class="preset-info-desc">${sample.desc}</div>
        </div>
        <span class="preset-scene-badge">${sample.scene}</span>
      `;

      btn.addEventListener('click', () => {
        document.querySelectorAll('.lidar-preset-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentSampleId = sample.id;
        runLidarAnalysis({ sample_id: sample.id });
      });

      dom.presetsList.appendChild(btn);
    });
  }

  /**
   * Handles user file upload (.pcd.bin / .bin)
   */
  async function uploadAndAnalyzeFile(file) {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    if (dom.loadedFilename) dom.loadedFilename.textContent = file.name;
    if (dom.loadedPointsCount) dom.loadedPointsCount.textContent = 'Uploading...';
    if (dom.loadedCard) dom.loadedCard.style.display = 'flex';

    // Deselect preset buttons
    document.querySelectorAll('.lidar-preset-btn').forEach((b) => b.classList.remove('active'));

    await executeLidarRequest('/api/lidar/analyze?score_thresh=' + state.scoreThreshold, {
      method: 'POST',
      body: formData
    });
  }

  /**
   * Executes LiDAR Perception & Autonomous Decision via REST API
   */
  async function runLidarAnalysis(params) {
    if (dom.btnRunLidar) {
      dom.btnRunLidar.disabled = true;
      dom.btnRunLidar.innerHTML = '<span class="spinner-sm"></span> Processing 3D LiDAR...';
    }

    const payload = {
      score_thresh: state.scoreThreshold,
      ...params
    };

    await executeLidarRequest('/api/lidar/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (dom.btnRunLidar) {
      dom.btnRunLidar.disabled = false;
      dom.btnRunLidar.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>Run LiDAR 3D Perception & Decision</span>
      `;
    }
  }

  /**
   * Core Fetch and Response Processor
   */
  async function executeLidarRequest(url, options) {
    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('LiDAR Analysis Error: ' + (err.error || 'Server error occurred'));
        return;
      }

      const data = await res.json();
      state.analysisData = data;

      // Update UI Header Stats & Loaded card
      if (dom.loadedFilename) dom.loadedFilename.textContent = data.sample_name || 'point_cloud.pcd.bin';
      if (dom.loadedPointsCount) dom.loadedPointsCount.textContent = `${(data.raw_point_count || 0).toLocaleString()} pts`;
      if (dom.loadedCard) dom.loadedCard.style.display = 'flex';

      // Update HUD
      if (dom.hudPoints) dom.hudPoints.textContent = `Pts: ${(data.rendered_points_count || 0).toLocaleString()} / ${(data.raw_point_count || 0).toLocaleString()}`;
      if (dom.hudLatency) dom.hudLatency.textContent = `Latency: ${data.inference_time_ms || 0} ms (CPU)`;
      if (dom.hudBoxes) dom.hudBoxes.textContent = `Detections: ${data.predictions_count || 0}`;

      // Update Decision Banner
      updateAutonomousDecisionBanner(data.decision, data.predictions);

      // Render 3D and 2D Visualizations
      renderFilteredViews();
    } catch (e) {
      console.error('[LiDAR] Request failed:', e);
      alert('Network or server connection failed while running LiDAR perception.');
    }
  }

  /**
   * Updates the Autonomous Decision Banner based strictly on LiDAR readings
   */
  function updateAutonomousDecisionBanner(decision, predictions) {
    if (!decision) return;

    // Action Badge
    if (dom.decisionBadge) {
      dom.decisionBadge.style.backgroundColor = decision.action_color || '#10b981';
      dom.decisionBadge.style.boxShadow = `0 4px 14px ${decision.action_color}44`;

      const iconMap = {
        EMERGENCY_STOP: '🛑',
        SLOW_DOWN: '⚠️',
        MOVE: '🟢',
        CHANGE_LEFT: '↔️',
        HOLD_LANE: '🔒'
      };

      const icon = iconMap[decision.final_decision] || '🛡️';
      dom.decisionBadge.innerHTML = `<span>${icon}</span> <span>${decision.final_decision}</span>`;
    }

    // Risk Score & Level
    if (dom.decisionRiskScore) {
      dom.decisionRiskScore.textContent = `${decision.risk_score.toFixed(1)} / 10`;
      dom.decisionRiskScore.style.color = decision.risk_score >= 8 ? '#ef4444' : decision.risk_score >= 5 ? '#f59e0b' : '#10b981';
    }

    if (dom.decisionRiskLevel) {
      dom.decisionRiskLevel.textContent = decision.risk_level;
      dom.decisionRiskLevel.className = `lidar-stat-tag ${decision.risk_level.toLowerCase()}`;
    }

    // Safe Target Speed
    if (dom.decisionSpeed) {
      dom.decisionSpeed.textContent = `${decision.safe_speed_kmh.toFixed(0)} km/h`;
    }

    // Closest Obstacle Distance
    if (dom.decisionClosestDist) {
      if (decision.closest_obstacle_dist !== null && decision.closest_obstacle_dist !== undefined) {
        dom.decisionClosestDist.textContent = `${decision.closest_obstacle_dist.toFixed(1)}m (${(decision.closest_obstacle_class || '').toUpperCase()})`;
      } else {
        dom.decisionClosestDist.textContent = 'None (>48m)';
      }
    }

    // Autonomous Decision Reason / Rationale
    if (dom.decisionReason) {
      dom.decisionReason.textContent = decision.decision_reason;
    }

    // Spatial Zones Telemetry
    if (dom.zoneFront) {
      if (decision.front_path_clear) {
        dom.zoneFront.textContent = 'CLEAR';
        dom.zoneFront.className = 'zone-status clear';
      } else {
        dom.zoneFront.textContent = 'OBSTRUCTED';
        dom.zoneFront.className = 'zone-status hazard';
      }
    }

    if (dom.zoneLeft) {
      const bs = decision.blind_spots || {};
      if (bs.left_occupied) {
        dom.zoneLeft.textContent = `OCCUPIED (${bs.left_closest_m ? bs.left_closest_m.toFixed(1) + 'm' : ''})`;
        dom.zoneLeft.className = 'zone-status occupied';
      } else {
        dom.zoneLeft.textContent = 'CLEAR';
        dom.zoneLeft.className = 'zone-status clear';
      }
    }

    if (dom.zoneRight) {
      const bs = decision.blind_spots || {};
      if (bs.right_occupied) {
        dom.zoneRight.textContent = `OCCUPIED (${bs.right_closest_m ? bs.right_closest_m.toFixed(1) + 'm' : ''})`;
        dom.zoneRight.className = 'zone-status occupied';
      } else {
        dom.zoneRight.textContent = 'CLEAR';
        dom.zoneRight.className = 'zone-status clear';
      }
    }
  }

  /**
   * Filter predictions based on active class filter and threshold
   */
  function getFilteredPredictions() {
    if (!state.analysisData || !state.analysisData.predictions) return [];
    return state.analysisData.predictions.filter((p) => {
      if (p.confidence < state.scoreThreshold) return false;
      if (state.activeFilterClass === 'ALL') return true;
      return p.class.toLowerCase() === state.activeFilterClass.toLowerCase();
    });
  }

  /**
   * Renders both 3D WebGL and 2D Tactical views and detections table
   */
  function renderFilteredViews() {
    const filtered = getFilteredPredictions();

    // Update Summary Stats
    const cars = filtered.filter((p) => p.class === 'car').length;
    const peds = filtered.filter((p) => p.class === 'pedestrian').length;

    if (dom.statTotalDetections) dom.statTotalDetections.textContent = filtered.length;
    if (dom.statCars) dom.statCars.textContent = cars;
    if (dom.statPedestrians) dom.statPedestrians.textContent = peds;
    if (dom.statLatency) dom.statLatency.textContent = `${state.analysisData?.inference_time_ms || 0} ms`;

    // Render Detections Cards
    renderDetectionsList(filtered);

    // Update 3D WebGL Points & Boxes
    updateThreeJsScene(filtered);

    // Update 2D BEV Canvas
    drawBevView(filtered);
  }

  /**
   * Renders the cards list of detected objects
   */
  function renderDetectionsList(predictions) {
    if (!dom.detectionsList) return;
    dom.detectionsList.innerHTML = '';

    if (!predictions.length) {
      dom.detectionsList.innerHTML = `
        <div class="lidar-empty-state">
          <svg class="lidar-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 8v4M12 16h.01"/>
          </svg>
          <div style="font-weight: 700; color: #475569;">No Objects Detected</div>
          <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;">Try lowering the confidence threshold slider or selecting another sample.</div>
        </div>
      `;
      return;
    }

    predictions.forEach((obj, idx) => {
      const card = document.createElement('div');
      card.className = `lidar-det-card ${state.hoveredObjIndex === idx ? 'hovered' : ''}`;
      card.dataset.index = idx;

      const isCar = obj.class === 'car';
      const c = obj.center || {};
      const d = obj.dimensions || {};
      const dist = obj.distance ? obj.distance.toFixed(1) : '0.0';
      const conf = (obj.confidence * 100).toFixed(0);

      // Determine hazard zone
      let zoneTag = 'Peripheral';
      let zoneClass = '';
      if (c.y >= 0 && c.y <= 35 && Math.abs(c.x) <= 2.0) {
        zoneTag = 'Front Corridor';
        zoneClass = 'front';
      } else if (c.x >= -4.5 && c.x <= -1.25 && c.y >= -8.0 && c.y <= 3.0) {
        zoneTag = 'Left Blind Spot';
        zoneClass = 'blindspot';
      } else if (c.x >= 1.25 && c.x <= 4.5 && c.y >= -8.0 && c.y <= 3.0) {
        zoneTag = 'Right Blind Spot';
        zoneClass = 'blindspot';
      }

      card.innerHTML = `
        <div class="det-card-col-left">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span class="lidar-det-badge ${isCar ? 'car' : 'pedestrian'}">
              ${isCar ? '🚗 CAR' : '🚶 PEDESTRIAN'}
            </span>
            <span style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700; color: #0284c7;">
              ${conf}%
            </span>
            <span class="lidar-det-zone ${zoneClass}">${zoneTag}</span>
          </div>
          <div class="lidar-det-coords">
            Pos: [X:${(c.x || 0).toFixed(1)}m, Y:${(c.y || 0).toFixed(1)}m, Z:${(c.z || 0).toFixed(1)}m] • Dim: [${(d.length || 0).toFixed(1)}×${(d.width || 0).toFixed(1)}×${(d.height || 0).toFixed(1)}m]
          </div>
        </div>
        <div class="det-card-col-right" style="text-align: right;">
          <div class="lidar-det-dist">${dist}m</div>
          <div style="font-size: 0.68rem; color: #94a3b8; font-family: var(--font-mono);">
            Yaw: ${((obj.yaw || 0) * 180 / Math.PI).toFixed(0)}°
          </div>
        </div>
      `;

      // Interactive card hover highlights box in 3D and 2D
      card.addEventListener('mouseenter', () => {
        state.hoveredObjIndex = idx;
        card.classList.add('hovered');
        highlightBoxInScenes(idx);
      });

      card.addEventListener('mouseleave', () => {
        state.hoveredObjIndex = null;
        card.classList.remove('hovered');
        highlightBoxInScenes(null);
      });

      dom.detectionsList.appendChild(card);
    });
  }

  /**
   * Highlights specific 3D box and 2D BEV item on hover
   */
  function highlightBoxInScenes(idx) {
    if (boxesGroup) {
      boxesGroup.children.forEach((mesh, i) => {
        if (mesh.userData && mesh.userData.index !== undefined) {
          const isTarget = mesh.userData.index === idx;
          mesh.material.opacity = isTarget ? 0.9 : 0.4;
          if (mesh.userData.line) {
            mesh.userData.line.material.linewidth = isTarget ? 3 : 1;
          }
        }
      });
    }
    drawBevView(getFilteredPredictions());
  }

  // ==============================================================================
  // THREE.JS 3D WEBGL POINT CLOUD & BOUNDING BOX RENDERER
  // ==============================================================================

  /**
   * Initializes the Three.js 3D Viewport
   */
  function initThreeJs() {
    if (!dom.threeViewport) return;

    const width = dom.threeViewport.clientWidth || 800;
    const height = dom.threeViewport.clientHeight || 450;

    // 1. Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14); // Deep Navy/Space Black
    scene.fog = new THREE.FogExp2(0x070b14, 0.015);

    // 2. Camera
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 300);
    setCameraView('isometric');

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = false;
    dom.threeViewport.appendChild(renderer.domElement);

    // 4. OrbitControls
    if (typeof THREE.OrbitControls !== 'undefined') {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't flip below ground
      controls.minDistance = 3;
      controls.maxDistance = 120;
      controls.target.set(0, 0, -10); // Look forward in front of the bus
    }

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x60a5fa, 0.6);
    dirLight.position.set(20, 40, 20);
    scene.add(dirLight);

    // 6. Ground Grid & Coordinate Helpers
    // In nuScenes LIDAR_TOP, ground is at Z ≈ -1.84m (mapped to Three.js Y = -1.84)
    gridHelper = new THREE.GridHelper(100, 50, 0x1e3a8a, 0x111c38);
    gridHelper.position.y = -1.84;
    scene.add(gridHelper);

    // 7. Ego Bus 3D Geometry
    createEgoBusMesh();

    // 8. Container Groups
    boxesGroup = new THREE.Group();
    scene.add(boxesGroup);

    // 9. Animation Loop
    animate();
  }

  /**
   * Builds 3D Ego Bus Avatar with Dimensions & Sensor Head
   */
  function createEgoBusMesh() {
    busGroup = new THREE.Group();

    // Standard Urban Transit Bus: Width 2.55m, Length 10.5m, Height 3.2m
    // LiDAR sensor is mounted on the roof forward: Y ≈ 1.0m, Z ≈ 0m relative to LiDAR center
    // In Three.js: X = Right, Y = Up, Z = Forward (-Z)
    const busGeo = new THREE.BoxGeometry(2.55, 3.2, 10.5);
    const busMat = new THREE.MeshBasicMaterial({
      color: 0x1d4ed8,
      wireframe: true,
      transparent: true,
      opacity: 0.6
    });
    const busMesh = new THREE.Mesh(busGeo, busMat);
    // Center the bus: roof is near 0, ground at -1.84, so center Y is at -0.24
    // Sensor is 2m from front bumper, so center Z is at +3.25
    busMesh.position.set(0, -0.24, 3.25);
    busGroup.add(busMesh);

    // Solid inner translucent cabin
    const cabinGeo = new THREE.BoxGeometry(2.5, 3.15, 10.45);
    const cabinMat = new THREE.MeshLambertMaterial({
      color: 0x0f172a,
      transparent: true,
      opacity: 0.75
    });
    const cabinMesh = new THREE.Mesh(cabinGeo, cabinMat);
    cabinMesh.position.copy(busMesh.position);
    busGroup.add(cabinMesh);

    // Forward Direction Arrow (LiDAR sensor origin pointing forward -Z)
    const arrowDir = new THREE.Vector3(0, 0, -1);
    const arrowOrigin = new THREE.Vector3(0, 0.2, 0);
    const arrowHelper = new THREE.ArrowHelper(arrowDir, arrowOrigin, 4.0, 0x38bdf8, 0.8, 0.4);
    busGroup.add(arrowHelper);

    // Sensor Beacon Pulse Ring
    const sensorGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.3, 16);
    const sensorMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    const sensorMesh = new THREE.Mesh(sensorGeo, sensorMat);
    sensorMesh.position.set(0, 0, 0);
    busGroup.add(sensorMesh);

    scene.add(busGroup);
  }

  /**
   * Sets camera view mode presets
   */
  function setCameraView(mode) {
    state.activeViewMode = mode;
    if (!camera) return;

    if (mode === 'top') {
      // Top-Down Bird's-Eye View
      camera.position.set(0, 60, -10);
      if (controls) {
        controls.target.set(0, 0, -10);
        controls.update();
      }
    } else if (mode === 'driver') {
      // Driver's Cabin Forward Perspective
      camera.position.set(0, 0.5, 2.0);
      if (controls) {
        controls.target.set(0, 0, -25);
        controls.update();
      }
    } else {
      // Default Isometric Chase View
      camera.position.set(-18, 22, 22);
      if (controls) {
        controls.target.set(0, 0, -10);
        controls.update();
      }
    }
  }

  /**
   * Three.js Animation Loop
   */
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  /**
   * Updates 3D Point Cloud and 3D Bounding Boxes
   */
  function updateThreeJsScene(predictions) {
    if (!scene) return;

    // 1. Update Point Cloud
    if (state.analysisData && state.analysisData.points) {
      const rawPoints = state.analysisData.points;
      const count = rawPoints.length;

      if (pointCloudMesh) {
        scene.remove(pointCloudMesh);
        if (pointCloudMesh.geometry) pointCloudMesh.geometry.dispose();
        if (pointCloudMesh.material) pointCloudMesh.material.dispose();
      }

      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);

      for (let i = 0; i < count; i++) {
        const pt = rawPoints[i];
        const lx = pt[0];
        const ly = pt[1];
        const lz = pt[2];
        const intensity = pt.length > 3 ? pt[3] : 15;

        // Coordinate mapping:
        // LiDAR +X (Right)   -> Three.js +X
        // LiDAR +Z (Up)      -> Three.js +Y
        // LiDAR +Y (Forward) -> Three.js -Z
        positions[i * 3] = lx;
        positions[i * 3 + 1] = lz;
        positions[i * 3 + 2] = -ly;

        // Height & Intensity Neon Shader Color
        // Ground (-1.84) -> Dark Teal/Blue; Objects (> -0.5) -> Vivid Cyan/Yellow/White
        const normZ = Math.min(1.0, Math.max(0.0, (lz + 2.0) / 4.0));
        let r = 0.1 + normZ * 0.7;
        let g = 0.3 + normZ * 0.7;
        let b = 0.7 + (1.0 - normZ) * 0.3;

        // Boost intensity
        if (intensity > 40) {
          r = Math.min(1.0, r + 0.3);
          g = Math.min(1.0, g + 0.3);
          b = 1.0;
        }

        colors[i * 3] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: 0.18,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        sizeAttenuation: true
      });

      pointCloudMesh = new THREE.Points(geometry, material);
      pointCloudMesh.visible = state.showPoints;
      scene.add(pointCloudMesh);
    }

    // 2. Update 3D Oriented Bounding Boxes
    if (boxesGroup) {
      // Clear previous boxes
      while (boxesGroup.children.length > 0) {
        const child = boxesGroup.children[0];
        boxesGroup.remove(child);
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      }

      predictions.forEach((obj, idx) => {
        const isCar = obj.class === 'car';
        const colorHex = isCar ? 0x2563eb : 0xec4899; // Electric Blue for Car, Hot Pink for Pedestrian
        const c = obj.center || {};
        const d = obj.dimensions || {};
        const yaw = obj.yaw || 0.0;

        // NuScenes LiDAR box conventions:
        // d.width along X, d.length along heading Y, d.height along Z
        // In Three.js: BoxGeometry(width, height, length)
        const bw = Math.max(0.4, d.width || 1.8);
        const bh = Math.max(0.5, d.height || 1.6);
        const bl = Math.max(0.5, d.length || 4.2);

        const boxGeo = new THREE.BoxGeometry(bw, bh, bl);

        // Translucent solid face
        const boxMat = new THREE.MeshBasicMaterial({
          color: colorHex,
          transparent: true,
          opacity: 0.25,
          depthWrite: false
        });
        const boxMesh = new THREE.Mesh(boxGeo, boxMat);

        // Position: X -> X, Z -> Y, Y -> -Z
        boxMesh.position.set(c.x || 0, c.z || 0, -(c.y || 0));

        // Rotation: Yaw around LiDAR Z -> Rotation around Three.js Y
        // Counter-clockwise around +Z maps directly to counter-clockwise around +Y!
        boxMesh.rotation.y = yaw;

        // Wireframe edges outline for sharp contrast
        const edges = new THREE.EdgesGeometry(boxGeo);
        const lineMat = new THREE.LineBasicMaterial({
          color: colorHex,
          linewidth: 2,
          transparent: true,
          opacity: 0.95
        });
        const wireframe = new THREE.LineSegments(edges, lineMat);
        boxMesh.add(wireframe);

        // Heading Arrow Pointer on top of box
        const headingDir = new THREE.Vector3(0, 0, -1);
        const headingOrigin = new THREE.Vector3(0, bh / 2 + 0.1, 0);
        const headingArrow = new THREE.ArrowHelper(headingDir, headingOrigin, bl * 0.45, 0xffffff, 0.4, 0.2);
        boxMesh.add(headingArrow);

        // Store index reference for hover
        boxMesh.userData = { index: idx, line: wireframe };

        boxesGroup.add(boxMesh);
      });

      boxesGroup.visible = state.showBoxes;
    }
  }

  function onWindowResize() {
    if (!renderer || !camera || !dom.threeViewport) return;
    const width = dom.threeViewport.clientWidth || 800;
    const height = dom.threeViewport.clientHeight || 450;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);

    if (bevCanvas) {
      initBevCanvas();
      drawBevView(getFilteredPredictions());
    }
  }

  // ==============================================================================
  // 2D TACTICAL BIRD'S-EYE VIEW (BEV) CANVAS
  // ==============================================================================

  /**
   * Initializes the 2D Tactical BEV Canvas
   */
  function initBevCanvas() {
    bevCanvas = dom.bevCanvas;
    if (!bevCanvas) return;

    const rect = bevCanvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    bevCanvas.width = rect.width * dpr;
    bevCanvas.height = rect.height * dpr;
    bevCtx = bevCanvas.getContext('2d');
    bevCtx.scale(dpr, dpr);
  }

  /**
   * Draws the 2D Tactical Radar Display with Oriented Boxes & Hazard Corridors
   */
  function drawBevView(predictions = []) {
    if (!bevCanvas || !bevCtx) return;

    const w = bevCanvas.clientWidth;
    const h = bevCanvas.clientHeight;
    bevCtx.clearRect(0, 0, w, h);

    // Coordinate Mapping:
    // Sensor Origin is placed at (w / 2, h * 0.82)
    // Forward (+Y) points UP on canvas (-canvas Y)
    // Right (+X) points RIGHT on canvas (+canvas X)
    // Scale: e.g. 50m forward spans ~ (h * 0.72)
    const originX = w / 2;
    const originY = h * 0.85;
    const scale = (h * 0.75) / 50.0; // pixels per meter

    function toCanvas(lx, ly) {
      return {
        x: originX + lx * scale,
        y: originY - ly * scale
      };
    }

    // 1. Draw Range Rings (10m, 20m, 30m, 40m, 50m)
    const rings = [10, 20, 30, 40, 50];
    rings.forEach((r) => {
      const radiusPx = r * scale;
      bevCtx.beginPath();
      bevCtx.arc(originX, originY, radiusPx, Math.PI, 2 * Math.PI, false);
      bevCtx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
      bevCtx.lineWidth = 1;
      bevCtx.setLineDash([4, 4]);
      bevCtx.stroke();
      bevCtx.setLineDash([]);

      // Range text
      bevCtx.fillStyle = 'rgba(148, 163, 184, 0.6)';
      bevCtx.font = '10px monospace';
      bevCtx.fillText(`${r}m`, originX + 6, originY - radiusPx + 12);
    });

    // 2. Draw Safety Corridors
    // Front Corridor: |X| <= 2.0m, Y in [0, 35m]
    const fcTopLeft = toCanvas(-2.0, 35.0);
    const fcBottomRight = toCanvas(2.0, 0.0);
    bevCtx.fillStyle = 'rgba(245, 158, 11, 0.05)';
    bevCtx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    bevCtx.lineWidth = 1.5;
    bevCtx.setLineDash([3, 3]);
    bevCtx.fillRect(fcTopLeft.x, fcTopLeft.y, fcBottomRight.x - fcTopLeft.x, fcBottomRight.y - fcTopLeft.y);
    bevCtx.strokeRect(fcTopLeft.x, fcTopLeft.y, fcBottomRight.x - fcTopLeft.x, fcBottomRight.y - fcTopLeft.y);
    bevCtx.setLineDash([]);

    // Blind Spots: Left [-4.5, -1.25] and Right [1.25, 4.5], Y [-8, 3]
    const bsLeftTL = toCanvas(-4.5, 3.0);
    const bsLeftBR = toCanvas(-1.25, -8.0);
    bevCtx.fillStyle = 'rgba(99, 102, 241, 0.06)';
    bevCtx.strokeStyle = 'rgba(99, 102, 241, 0.35)';
    bevCtx.setLineDash([2, 2]);
    bevCtx.fillRect(bsLeftTL.x, bsLeftTL.y, bsLeftBR.x - bsLeftTL.x, bsLeftBR.y - bsLeftTL.y);
    bevCtx.strokeRect(bsLeftTL.x, bsLeftTL.y, bsLeftBR.x - bsLeftTL.x, bsLeftBR.y - bsLeftTL.y);

    const bsRightTL = toCanvas(1.25, 3.0);
    const bsRightBR = toCanvas(4.5, -8.0);
    bevCtx.fillRect(bsRightTL.x, bsRightTL.y, bsRightBR.x - bsRightTL.x, bsRightBR.y - bsRightTL.y);
    bevCtx.strokeRect(bsRightTL.x, bsRightTL.y, bsRightBR.x - bsRightTL.x, bsRightBR.y - bsRightTL.y);
    bevCtx.setLineDash([]);

    // 3. Draw 2D Points (if enabled)
    if (state.showPoints && state.analysisData && state.analysisData.points) {
      const pts = state.analysisData.points;
      bevCtx.fillStyle = 'rgba(56, 189, 248, 0.4)';
      for (let i = 0; i < pts.length; i += 2) {
        const cp = toCanvas(pts[i][0], pts[i][1]);
        if (cp.x >= 0 && cp.x <= w && cp.y >= 0 && cp.y <= h) {
          bevCtx.fillRect(cp.x, cp.y, 1.2, 1.2);
        }
      }
    }

    // 4. Draw Ego Bus Shape
    // Bus width 2.5m, length 10.5m; center around sensor: Y extends +2.0m forward, -8.5m rear
    const busTL = toCanvas(-1.25, 2.0);
    const busBR = toCanvas(1.25, -8.5);
    bevCtx.fillStyle = 'rgba(29, 78, 216, 0.85)';
    bevCtx.strokeStyle = '#60a5fa';
    bevCtx.lineWidth = 1.5;
    bevCtx.fillRect(busTL.x, busTL.y, busBR.x - busTL.x, busBR.y - busTL.y);
    bevCtx.strokeRect(busTL.x, busTL.y, busBR.x - busTL.x, busBR.y - busTL.y);

    // Front Bumper Marker
    bevCtx.fillStyle = '#38bdf8';
    bevCtx.fillRect(busTL.x, busTL.y - 2, busBR.x - busTL.x, 3);

    // Bus Label
    bevCtx.fillStyle = '#ffffff';
    bevCtx.font = 'bold 9px sans-serif';
    bevCtx.textAlign = 'center';
    bevCtx.fillText('AUTO BUS', originX, originY - 10);

    // 5. Draw 2D Oriented Detections
    if (state.showBoxes) {
      predictions.forEach((obj, idx) => {
        const isCar = obj.class === 'car';
        const isHovered = state.hoveredObjIndex === idx;
        const color = isCar ? '#2563eb' : '#ec4899';
        const c = obj.center || {};
        const d = obj.dimensions || {};
        const yaw = obj.yaw || 0.0;

        const cp = toCanvas(c.x, c.y);
        const bw = Math.max(8, (d.width || 1.8) * scale);
        const bl = Math.max(10, (d.length || 4.2) * scale);

        bevCtx.save();
        bevCtx.translate(cp.x, cp.y);
        // Canvas rotation: yaw is counter-clockwise in LiDAR (where +Y is up, +X is right)
        // Canvas Y is inverted, so rotation angle is -yaw
        bevCtx.rotate(-yaw);

        bevCtx.fillStyle = isHovered ? (isCar ? 'rgba(37,99,235,0.7)' : 'rgba(236,72,153,0.7)') : (isCar ? 'rgba(37,99,235,0.35)' : 'rgba(236,72,153,0.35)');
        bevCtx.strokeStyle = isHovered ? '#ffffff' : color;
        bevCtx.lineWidth = isHovered ? 2.5 : 1.5;

        // Rectangle centered at (0, 0)
        bevCtx.fillRect(-bw / 2, -bl / 2, bw, bl);
        bevCtx.strokeRect(-bw / 2, -bl / 2, bw, bl);

        // Heading arrow pointing forward on local box (-Y on canvas)
        bevCtx.beginPath();
        bevCtx.moveTo(0, 0);
        bevCtx.lineTo(0, -bl / 2);
        bevCtx.strokeStyle = '#ffffff';
        bevCtx.lineWidth = 1.5;
        bevCtx.stroke();

        bevCtx.restore();

        // Distance & Class text badge
        bevCtx.fillStyle = isHovered ? '#ffffff' : '#cbd5e1';
        bevCtx.font = 'bold 9px monospace';
        bevCtx.textAlign = 'center';
        bevCtx.fillText(`${isCar ? 'Car' : 'Ped'} ${(obj.distance || 0).toFixed(1)}m`, cp.x, cp.y - bl / 2 - 4);
      });
    }
  }

})();
