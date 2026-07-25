/**
 * Satellite model with subsystems and fault effects
 * Enhanced with detailed geometry, lights, and audio feedback
 */

import * as THREE from 'three';

export interface SatelliteState {
    telemetry: {
        battery_soc: number;
        cpu_temperature: number;
        power_temperature: number;
        cpu_load: number;
        current_draw: number;
        memory_integrity: number;
        signal_stability: number;
        voltage: number;
    };
    faults: {
        seu: boolean;
        latchup: boolean;
        thermal: boolean;
        memory: boolean;
        power: boolean;
    };
    action: number | null;
    step: number;
    episode: number;
    totalReward: number;
    faultsRecovered: number;
}

export interface Satellite {
    group: THREE.Group;
    body: THREE.Mesh;
    solarPanelLeft: THREE.Mesh;
    solarPanelRight: THREE.Mesh;
    antenna: THREE.Mesh;
    cpuModule: THREE.Mesh;
    powerModule: THREE.Mesh;
    faultEffects: {
        seu: THREE.Points;
        thermal: THREE.PointLight;
        power: THREE.PointLight;
        memory: THREE.PointLight;
        latchup: THREE.PointLight;
        recoveryGlow: THREE.PointLight;
    };
    statusLights: {
        normal: THREE.Mesh;
        warning: THREE.Mesh;
        critical: THREE.Mesh;
    };
    audioContext: AudioContext | null;
    sounds: {
        faultAlarm: OscillatorNode | null;
        recoveryChime: OscillatorNode | null;
        gainNode: GainNode | null;
    };
    prevFaults: Record<string, boolean>;
    prevAction: number | null;
    powerShutdownActive: boolean;
    shutdownStartTime: number;
}

// Audio helper functions
function createAudioContext(): AudioContext | null {
    try {
        return new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch (e) {
        console.warn('Audio not supported');
        return null;
    }
}

// Play SEU fault sound - digital glitch/static burst
function playSEUSound(ctx: AudioContext): void {
    const duration = 0.4;
    const now = ctx.currentTime;
    
    // Create noise buffer for glitch effect
    const bufferSize = ctx.sampleRate * duration;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }
    
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    
    // High-pass filter for digital sound
    const filter = ctx.createBiquadFilter();
    filter.type = 'highpass';
    filter.frequency.value = 2000;
    
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + duration);
    
    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    
    noise.start(now);
    noise.stop(now + duration);
    
    // Add high-pitched beep
    const beep = ctx.createOscillator();
    beep.type = 'square';
    beep.frequency.setValueAtTime(1200, now);
    beep.frequency.setValueAtTime(800, now + 0.1);
    beep.frequency.setValueAtTime(1500, now + 0.2);
    
    const beepGain = ctx.createGain();
    beepGain.gain.setValueAtTime(0.08, now);
    beepGain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
    
    beep.connect(beepGain);
    beepGain.connect(ctx.destination);
    beep.start(now);
    beep.stop(now + 0.3);
}

// Play Latch-up sound - heavy electrical surge
function playLatchupSound(ctx: AudioContext): void {
    const now = ctx.currentTime;
    
    // Deep rumble
    const osc1 = ctx.createOscillator();
    osc1.type = 'sawtooth';
    osc1.frequency.setValueAtTime(80, now);
    osc1.frequency.exponentialRampToValueAtTime(40, now + 0.8);
    
    // Electrical crackle
    const osc2 = ctx.createOscillator();
    osc2.type = 'square';
    osc2.frequency.setValueAtTime(150, now);
    osc2.frequency.setValueAtTime(200, now + 0.2);
    osc2.frequency.setValueAtTime(100, now + 0.4);
    
    const gain1 = ctx.createGain();
    gain1.gain.setValueAtTime(0.25, now);
    gain1.gain.exponentialRampToValueAtTime(0.01, now + 0.8);
    
    const gain2 = ctx.createGain();
    gain2.gain.setValueAtTime(0.1, now);
    gain2.gain.setValueAtTime(0.15, now + 0.1);
    gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.6);
    
    // Distortion for harsh sound
    const distortion = ctx.createWaveShaper();
    const curve = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
        const x = (i / 128) - 1;
        curve[i] = Math.tanh(x * 3);
    }
    distortion.curve = curve;
    
    osc1.connect(distortion);
    distortion.connect(gain1);
    gain1.connect(ctx.destination);
    
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    
    osc1.start(now);
    osc1.stop(now + 0.8);
    osc2.start(now);
    osc2.stop(now + 0.6);
}

// Play Thermal fault sound - rising alarm tone
function playThermalSound(ctx: AudioContext): void {
    const now = ctx.currentTime;
    
    // Rising warning tone
    const osc = ctx.createOscillator();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(300, now);
    osc.frequency.linearRampToValueAtTime(600, now + 0.3);
    osc.frequency.linearRampToValueAtTime(300, now + 0.6);
    osc.frequency.linearRampToValueAtTime(700, now + 0.9);
    
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.15, now);
    gain.gain.setValueAtTime(0.2, now + 0.3);
    gain.gain.setValueAtTime(0.15, now + 0.6);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 1.0);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start(now);
    osc.stop(now + 1.0);
    
    // Add hissing/steam sound
    const bufferSize = ctx.sampleRate * 0.8;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * 0.3 * (1 - i / bufferSize);
    }
    
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    
    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 4000;
    filter.Q.value = 2;
    
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.1, now);
    
    noise.connect(filter);
    filter.connect(noiseGain);
    noiseGain.connect(ctx.destination);
    
    noise.start(now);
    noise.stop(now + 0.8);
}

// Play Memory fault sound - corrupted data beeps
function playMemorySound(ctx: AudioContext): void {
    const now = ctx.currentTime;
    
    // Rapid corrupted beep sequence
    for (let i = 0; i < 6; i++) {
        const osc = ctx.createOscillator();
        osc.type = 'square';
        osc.frequency.value = 400 + Math.random() * 800;
        
        const gain = ctx.createGain();
        gain.gain.setValueAtTime(0.1, now + i * 0.08);
        gain.gain.exponentialRampToValueAtTime(0.01, now + i * 0.08 + 0.06);
        
        osc.connect(gain);
        gain.connect(ctx.destination);
        
        osc.start(now + i * 0.08);
        osc.stop(now + i * 0.08 + 0.06);
    }
    
    // Error tone
    const errorOsc = ctx.createOscillator();
    errorOsc.type = 'sine';
    errorOsc.frequency.setValueAtTime(880, now + 0.5);
    errorOsc.frequency.setValueAtTime(440, now + 0.6);
    
    const errorGain = ctx.createGain();
    errorGain.gain.setValueAtTime(0.12, now + 0.5);
    errorGain.gain.exponentialRampToValueAtTime(0.01, now + 0.8);
    
    errorOsc.connect(errorGain);
    errorGain.connect(ctx.destination);
    
    errorOsc.start(now + 0.5);
    errorOsc.stop(now + 0.8);
}

// Play POWER fault sound - EMERGENCY KLAXON (full shutdown alarm)
function playPowerEmergencySound(ctx: AudioContext): void {
    const now = ctx.currentTime;
    const duration = 2.5; // Longer emergency sound
    
    // Main klaxon - alternating two-tone alarm
    for (let i = 0; i < 5; i++) {
        const osc1 = ctx.createOscillator();
        osc1.type = 'sawtooth';
        osc1.frequency.value = 440; // A4
        
        const osc2 = ctx.createOscillator();
        osc2.type = 'sawtooth';
        osc2.frequency.value = 370; // F#4
        
        const gain1 = ctx.createGain();
        const gain2 = ctx.createGain();
        
        const t = now + i * 0.5;
        gain1.gain.setValueAtTime(0.25, t);
        gain1.gain.setValueAtTime(0, t + 0.25);
        
        gain2.gain.setValueAtTime(0, t);
        gain2.gain.setValueAtTime(0.25, t + 0.25);
        gain2.gain.setValueAtTime(0, t + 0.5);
        
        osc1.connect(gain1);
        osc2.connect(gain2);
        gain1.connect(ctx.destination);
        gain2.connect(ctx.destination);
        
        osc1.start(t);
        osc1.stop(t + 0.25);
        osc2.start(t + 0.25);
        osc2.stop(t + 0.5);
    }
    
    // Deep power-down rumble
    const rumble = ctx.createOscillator();
    rumble.type = 'sawtooth';
    rumble.frequency.setValueAtTime(100, now);
    rumble.frequency.exponentialRampToValueAtTime(20, now + duration);
    
    const rumbleGain = ctx.createGain();
    rumbleGain.gain.setValueAtTime(0.3, now);
    rumbleGain.gain.exponentialRampToValueAtTime(0.01, now + duration);
    
    // Low-pass for deep rumble
    const lowpass = ctx.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 200;
    
    rumble.connect(lowpass);
    lowpass.connect(rumbleGain);
    rumbleGain.connect(ctx.destination);
    
    rumble.start(now);
    rumble.stop(now + duration);
    
    // Electrical shutdown sound
    const shutdown = ctx.createOscillator();
    shutdown.type = 'sine';
    shutdown.frequency.setValueAtTime(1000, now + 0.5);
    shutdown.frequency.exponentialRampToValueAtTime(50, now + 1.5);
    
    const shutdownGain = ctx.createGain();
    shutdownGain.gain.setValueAtTime(0.15, now + 0.5);
    shutdownGain.gain.exponentialRampToValueAtTime(0.01, now + 1.5);
    
    shutdown.connect(shutdownGain);
    shutdownGain.connect(ctx.destination);
    
    shutdown.start(now + 0.5);
    shutdown.stop(now + 1.5);
}

function playFaultSound(satellite: Satellite, faultType: string): void {
    if (!satellite.audioContext) return;
    
    try {
        // Resume audio context if suspended (browser autoplay policy)
        if (satellite.audioContext.state === 'suspended') {
            satellite.audioContext.resume();
        }
        
        switch (faultType) {
            case 'seu':
                playSEUSound(satellite.audioContext);
                break;
            case 'latchup':
                playLatchupSound(satellite.audioContext);
                break;
            case 'thermal':
                playThermalSound(satellite.audioContext);
                break;
            case 'memory':
                playMemorySound(satellite.audioContext);
                break;
            case 'power':
                playPowerEmergencySound(satellite.audioContext);
                // Trigger shutdown sequence
                satellite.powerShutdownActive = true;
                satellite.shutdownStartTime = Date.now();
                break;
            default:
                playSEUSound(satellite.audioContext);
        }
    } catch (e) {
        console.warn('Audio playback error:', e);
    }
}

function playRecoverySound(satellite: Satellite): void {
    if (!satellite.audioContext) return;
    
    try {
        if (satellite.audioContext.state === 'suspended') {
            satellite.audioContext.resume();
        }
        
        const ctx = satellite.audioContext;
        const now = ctx.currentTime;
        
        // Triumphant recovery chime - ascending arpeggio
        const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
        
        notes.forEach((freq, i) => {
            const osc = ctx.createOscillator();
            osc.type = 'sine';
            osc.frequency.value = freq;
            
            const gain = ctx.createGain();
            gain.gain.setValueAtTime(0, now + i * 0.1);
            gain.gain.linearRampToValueAtTime(0.12, now + i * 0.1 + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.01, now + i * 0.1 + 0.3);
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start(now + i * 0.1);
            osc.stop(now + i * 0.1 + 0.3);
        });
        
        // Add shimmering overtone
        const shimmer = ctx.createOscillator();
        shimmer.type = 'triangle';
        shimmer.frequency.setValueAtTime(1046.50, now + 0.3);
        shimmer.frequency.setValueAtTime(1318.51, now + 0.5);
        
        const shimmerGain = ctx.createGain();
        shimmerGain.gain.setValueAtTime(0.08, now + 0.3);
        shimmerGain.gain.exponentialRampToValueAtTime(0.01, now + 0.7);
        
        shimmer.connect(shimmerGain);
        shimmerGain.connect(ctx.destination);
        
        shimmer.start(now + 0.3);
        shimmer.stop(now + 0.7);
    } catch (e) {
        // Ignore audio errors
    }
}

export function createSatellite(): Satellite {
    const group = new THREE.Group();
    // BIGGER satellite - increased scale from 0.05 to 0.15
    group.scale.setScalar(0.15);
    
    // Main body (bus) - more detailed with beveled edges
    const bodyGeometry = new THREE.BoxGeometry(1.2, 1.0, 1.5, 2, 2, 2);
    const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0x3a3a3a,
        metalness: 0.9,
        roughness: 0.2,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    group.add(body);
    
    // Gold foil thermal blanket effect with shimmer
    const foilGeometry = new THREE.BoxGeometry(1.22, 1.02, 1.52);
    const foilMaterial = new THREE.MeshStandardMaterial({
        color: 0xd4af37,
        metalness: 0.95,
        roughness: 0.1,
        transparent: true,
        opacity: 0.7,
        emissive: 0x4a3a00,
        emissiveIntensity: 0.15,
    });
    const foil = new THREE.Mesh(foilGeometry, foilMaterial);
    group.add(foil);
    
    // Decorative panel lines
    const linesMaterial = new THREE.MeshStandardMaterial({
        color: 0x222222,
        metalness: 0.5,
        roughness: 0.8,
    });
    for (let i = -0.4; i <= 0.4; i += 0.2) {
        const line = new THREE.Mesh(
            new THREE.BoxGeometry(1.25, 0.02, 0.02),
            linesMaterial
        );
        line.position.set(0, i, 0.77);
        group.add(line);
    }
    
    // Solar panels - larger and more detailed with cells
    const panelGeometry = new THREE.BoxGeometry(3.5, 0.08, 1.2);
    const panelMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a237e,
        metalness: 0.4,
        roughness: 0.3,
        emissive: 0x0d1545,
        emissiveIntensity: 0.3,
    });
    
    const solarPanelLeft = new THREE.Mesh(panelGeometry, panelMaterial);
    solarPanelLeft.position.x = -2.35;
    group.add(solarPanelLeft);
    
    // Solar cell grid on left panel
    const cellMaterial = new THREE.MeshStandardMaterial({
        color: 0x0a0a30,
        emissive: 0x1a1a60,
        emissiveIntensity: 0.4,
    });
    for (let x = -1.5; x <= 1.5; x += 0.4) {
        for (let z = -0.4; z <= 0.4; z += 0.3) {
            const cell = new THREE.Mesh(
                new THREE.BoxGeometry(0.35, 0.09, 0.25),
                cellMaterial
            );
            cell.position.set(x - 2.35, 0.01, z);
            group.add(cell);
        }
    }
    
    const solarPanelRight = new THREE.Mesh(panelGeometry, panelMaterial);
    solarPanelRight.position.x = 2.35;
    group.add(solarPanelRight);
    
    // Solar cell grid on right panel
    for (let x = -1.5; x <= 1.5; x += 0.4) {
        for (let z = -0.4; z <= 0.4; z += 0.3) {
            const cell = new THREE.Mesh(
                new THREE.BoxGeometry(0.35, 0.09, 0.25),
                cellMaterial
            );
            cell.position.set(x + 2.35, 0.01, z);
            group.add(cell);
        }
    }
    
    // Panel hinges/arms
    const armMaterial = new THREE.MeshStandardMaterial({
        color: 0x666666,
        metalness: 0.9,
        roughness: 0.3,
    });
    const leftArm = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.15, 0.15),
        armMaterial
    );
    leftArm.position.set(-0.9, 0, 0);
    group.add(leftArm);
    
    const rightArm = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 0.15, 0.15),
        armMaterial
    );
    rightArm.position.set(0.9, 0, 0);
    group.add(rightArm);
    
    // High-gain antenna dish - larger and more detailed
    const antennaGeometry = new THREE.ConeGeometry(0.6, 0.3, 32, 1, true);
    const antennaMaterial = new THREE.MeshStandardMaterial({
        color: 0xeeeeee,
        metalness: 0.95,
        roughness: 0.05,
        side: THREE.DoubleSide,
    });
    const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
    antenna.position.set(0, 0.7, 0);
    antenna.rotation.x = -Math.PI / 2;
    group.add(antenna);
    
    // Antenna feed
    const feedGeometry = new THREE.CylinderGeometry(0.05, 0.08, 0.2);
    const feed = new THREE.Mesh(feedGeometry, antennaMaterial);
    feed.position.set(0, 0.85, 0);
    group.add(feed);
    
    // Antenna stem with support struts
    const stemGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.4);
    const stem = new THREE.Mesh(stemGeometry, armMaterial);
    stem.position.set(0, 0.7, 0);
    group.add(stem);
    
    // CPU module (internal indicator) - enhanced
    const cpuGeometry = new THREE.BoxGeometry(0.4, 0.25, 0.4);
    const cpuMaterial = new THREE.MeshStandardMaterial({
        color: 0x00ff00,
        emissive: 0x00ff00,
        emissiveIntensity: 0.6,
        metalness: 0.3,
        roughness: 0.7,
    });
    const cpuModule = new THREE.Mesh(cpuGeometry, cpuMaterial);
    cpuModule.position.set(0.3, 0, 0.4);
    group.add(cpuModule);
    
    // Heat sink on CPU
    for (let i = -0.15; i <= 0.15; i += 0.06) {
        const fin = new THREE.Mesh(
            new THREE.BoxGeometry(0.02, 0.15, 0.35),
            armMaterial
        );
        fin.position.set(0.3 + i, 0.2, 0.4);
        group.add(fin);
    }
    
    // Power module - enhanced with battery cells look
    const powerGeometry = new THREE.BoxGeometry(0.4, 0.25, 0.4);
    const powerMaterial = new THREE.MeshStandardMaterial({
        color: 0x00aaff,
        emissive: 0x00aaff,
        emissiveIntensity: 0.4,
        metalness: 0.4,
        roughness: 0.6,
    });
    const powerModule = new THREE.Mesh(powerGeometry, powerMaterial);
    powerModule.position.set(-0.3, 0, 0.4);
    group.add(powerModule);
    
    // Thruster nozzles
    const thrusterMaterial = new THREE.MeshStandardMaterial({
        color: 0x444444,
        metalness: 0.8,
        roughness: 0.4,
    });
    const thrusterPositions = [
        [0.5, -0.4, 0.6],
        [-0.5, -0.4, 0.6],
        [0.5, -0.4, -0.6],
        [-0.5, -0.4, -0.6],
    ];
    thrusterPositions.forEach(([x, y, z]) => {
        const thruster = new THREE.Mesh(
            new THREE.ConeGeometry(0.08, 0.15, 8),
            thrusterMaterial
        );
        thruster.position.set(x, y, z);
        thruster.rotation.x = Math.PI;
        group.add(thruster);
    });
    
    // === STATUS INDICATOR LIGHTS ===
    const lightGeometry = new THREE.SphereGeometry(0.08, 16, 16);
    
    // Normal status light (green)
    const normalLightMat = new THREE.MeshStandardMaterial({
        color: 0x00ff00,
        emissive: 0x00ff00,
        emissiveIntensity: 1.0,
        transparent: true,
    });
    const normalLight = new THREE.Mesh(lightGeometry, normalLightMat);
    normalLight.position.set(0.5, 0.52, 0);
    group.add(normalLight);
    
    // Warning light (yellow/orange)
    const warningLightMat = new THREE.MeshStandardMaterial({
        color: 0xffaa00,
        emissive: 0xffaa00,
        emissiveIntensity: 0,
        transparent: true,
    });
    const warningLight = new THREE.Mesh(lightGeometry, warningLightMat);
    warningLight.position.set(0.3, 0.52, 0);
    group.add(warningLight);
    
    // Critical light (red)
    const criticalLightMat = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        emissive: 0xff0000,
        emissiveIntensity: 0,
        transparent: true,
    });
    const criticalLight = new THREE.Mesh(lightGeometry, criticalLightMat);
    criticalLight.position.set(0.1, 0.52, 0);
    group.add(criticalLight);
    
    // === FAULT EFFECTS ===
    
    // SEU sparkles - more particles
    const seuGeometry = new THREE.BufferGeometry();
    const seuPositions = new Float32Array(150 * 3);
    for (let i = 0; i < seuPositions.length; i++) {
        seuPositions[i] = (Math.random() - 0.5) * 2.5;
    }
    seuGeometry.setAttribute('position', new THREE.BufferAttribute(seuPositions, 3));
    const seuMaterial = new THREE.PointsMaterial({
        color: 0x00ffff,
        size: 0.08,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
    });
    const seuEffect = new THREE.Points(seuGeometry, seuMaterial);
    group.add(seuEffect);
    
    // Thermal glow - brighter
    const thermalLight = new THREE.PointLight(0xff4400, 0, 5);
    thermalLight.position.set(0, 0, 0);
    group.add(thermalLight);
    
    // Power surge light
    const powerLight = new THREE.PointLight(0xffff00, 0, 5);
    powerLight.position.set(-0.3, 0, 0.4);
    group.add(powerLight);
    
    // Memory fault light (purple/cyan)
    const memoryLight = new THREE.PointLight(0x8800ff, 0, 4);
    memoryLight.position.set(0.3, 0, 0.4);
    group.add(memoryLight);
    
    // Latchup light (intense red)
    const latchupLight = new THREE.PointLight(0xff0044, 0, 6);
    latchupLight.position.set(0, -0.3, 0);
    group.add(latchupLight);
    
    // Recovery glow (green pulse when RL takes action)
    const recoveryGlow = new THREE.PointLight(0x00ff88, 0, 8);
    recoveryGlow.position.set(0, 0, 0);
    group.add(recoveryGlow);
    
    // Initialize audio
    const audioContext = createAudioContext();
    
    return {
        group,
        body,
        solarPanelLeft,
        solarPanelRight,
        antenna,
        cpuModule,
        powerModule,
        faultEffects: {
            seu: seuEffect,
            thermal: thermalLight,
            power: powerLight,
            memory: memoryLight,
            latchup: latchupLight,
            recoveryGlow: recoveryGlow,
        },
        statusLights: {
            normal: normalLight,
            warning: warningLight,
            critical: criticalLight,
        },
        audioContext,
        sounds: {
            faultAlarm: null,
            recoveryChime: null,
            gainNode: null,
        },
        prevFaults: { seu: false, latchup: false, thermal: false, memory: false, power: false },
        prevAction: null,
        powerShutdownActive: false,
        shutdownStartTime: 0,
    };
}

export function updateSatellite(satellite: Satellite, state: SatelliteState, delta: number): void {
    const time = Date.now();
    
    // Check for NEW fault activations to play sounds
    const faultTypes = ['seu', 'latchup', 'thermal', 'memory', 'power'] as const;
    faultTypes.forEach(faultType => {
        if (state.faults[faultType] && !satellite.prevFaults[faultType]) {
            // NEW fault detected - play alarm sound
            playFaultSound(satellite, faultType);
        }
    });
    
    // Check for recovery action (action 1-6 and previous action was different)
    if (state.action !== null && state.action > 0 && state.action <= 6 && state.action !== satellite.prevAction) {
        playRecoverySound(satellite);
        // Flash recovery glow
        satellite.faultEffects.recoveryGlow.intensity = 5;
        
        // If power fault was active and now recovering, end shutdown sequence
        if (satellite.powerShutdownActive && (state.action === 4 || !state.faults.power)) {
            satellite.powerShutdownActive = false;
        }
    }
    
    // Store current state for next comparison
    satellite.prevFaults = { ...state.faults };
    satellite.prevAction = state.action;
    
    // === POWER SHUTDOWN SEQUENCE ===
    // Power fault = Complete system shutdown with dramatic visual effect
    const shutdownDuration = 3000; // 3 seconds for full shutdown
    const recoveryDuration = 2000; // 2 seconds for power-up
    
    let shutdownFactor = 1.0; // 1.0 = normal, 0.0 = fully shutdown
    
    if (satellite.powerShutdownActive) {
        const elapsed = time - satellite.shutdownStartTime;
        
        if (state.faults.power) {
            // Shutting down
            shutdownFactor = Math.max(0, 1 - (elapsed / shutdownDuration));
        } else {
            // Recovering (power-up sequence)
            const recoveryElapsed = elapsed - shutdownDuration;
            if (recoveryElapsed > 0) {
                shutdownFactor = Math.min(1, recoveryElapsed / recoveryDuration);
                if (shutdownFactor >= 1) {
                    satellite.powerShutdownActive = false;
                }
            } else {
                shutdownFactor = 0;
            }
        }
    } else if (state.faults.power && !satellite.prevFaults.power) {
        // Just started power fault
        satellite.powerShutdownActive = true;
        satellite.shutdownStartTime = time;
        shutdownFactor = 1.0;
    }
    
    // Apply shutdown factor to all emissive materials
    const bodyMat = satellite.body.material as THREE.MeshStandardMaterial;
    bodyMat.emissiveIntensity = shutdownFactor * 0.1;
    
    // Update CPU module color based on temperature (affected by shutdown)
    const cpuMat = satellite.cpuModule.material as THREE.MeshStandardMaterial;
    const cpuHeat = state.telemetry.cpu_temperature;
    cpuMat.emissiveIntensity = (0.4 + cpuHeat * 0.8) * shutdownFactor;
    cpuMat.emissive.setHSL(0.33 - cpuHeat * 0.33, 1, 0.5);
    cpuMat.color.setHSL(0.33 - cpuHeat * 0.33, 1, 0.5 * (0.3 + shutdownFactor * 0.7));
    
    // Update power module based on battery (affected by shutdown)
    const powerMat = satellite.powerModule.material as THREE.MeshStandardMaterial;
    const batteryLevel = state.telemetry.battery_soc;
    powerMat.emissiveIntensity = batteryLevel * 0.6 * shutdownFactor;
    powerMat.emissive.setHSL(0.55 + (1 - batteryLevel) * 0.1, 1, 0.4);
    
    // Solar panel gentle movement (tracking sun) - stops during shutdown
    if (shutdownFactor > 0.1) {
        satellite.solarPanelLeft.rotation.x = Math.sin(time / 3000) * 0.08 * shutdownFactor;
        satellite.solarPanelRight.rotation.x = -Math.sin(time / 3000) * 0.08 * shutdownFactor;
        satellite.solarPanelLeft.rotation.z = Math.sin(time / 5000) * 0.03 * shutdownFactor;
        satellite.solarPanelRight.rotation.z = -Math.sin(time / 5000) * 0.03 * shutdownFactor;
    }
    
    // Antenna slight wobble - stops during shutdown
    if (shutdownFactor > 0.1) {
        satellite.antenna.rotation.x = -Math.PI / 2 + Math.sin(time / 4000) * 0.02 * shutdownFactor;
        satellite.antenna.rotation.y = Math.sin(time / 3500) * 0.02 * shutdownFactor;
    }
    
    // === STATUS INDICATOR LIGHTS ===
    const activeFaultCount = Object.values(state.faults).filter(v => v).length;
    const normalMat = satellite.statusLights.normal.material as THREE.MeshStandardMaterial;
    const warningMat = satellite.statusLights.warning.material as THREE.MeshStandardMaterial;
    const criticalMat = satellite.statusLights.critical.material as THREE.MeshStandardMaterial;
    
    if (state.faults.power && shutdownFactor < 0.5) {
        // POWER FAULT - All lights flicker then go dark during shutdown
        const flicker = Math.random() > 0.7 ? 1 : 0;
        normalMat.emissiveIntensity = flicker * shutdownFactor * 0.3;
        warningMat.emissiveIntensity = flicker * shutdownFactor * 0.3;
        criticalMat.emissiveIntensity = (0.5 + Math.sin(time / 50) * 0.5) * shutdownFactor;
    } else if (activeFaultCount === 0) {
        // All normal - green light on, others off
        normalMat.emissiveIntensity = (0.8 + Math.sin(time / 500) * 0.2) * shutdownFactor;
        warningMat.emissiveIntensity = 0;
        criticalMat.emissiveIntensity = 0;
    } else if (activeFaultCount <= 2) {
        // Warning state - yellow flashing
        normalMat.emissiveIntensity = 0.2 * shutdownFactor;
        warningMat.emissiveIntensity = (0.5 + Math.sin(time / 200) * 0.5) * shutdownFactor;
        criticalMat.emissiveIntensity = 0;
    } else {
        // Critical state - red rapid flashing
        normalMat.emissiveIntensity = 0;
        warningMat.emissiveIntensity = 0.3 * shutdownFactor;
        criticalMat.emissiveIntensity = (0.5 + Math.sin(time / 80) * 0.5) * shutdownFactor;
    }
    
    // === FAULT VISUAL EFFECTS ===
    
    // SEU sparkles (cyan electric particles)
    const seuMat = satellite.faultEffects.seu.material as THREE.PointsMaterial;
    if (state.faults.seu) {
        seuMat.opacity = 0.9;
        seuMat.size = 0.1 + Math.sin(time / 50) * 0.03;
        const positions = satellite.faultEffects.seu.geometry.attributes.position.array as Float32Array;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] += (Math.random() - 0.5) * 0.15;
            positions[i + 1] += (Math.random() - 0.5) * 0.15;
            positions[i + 2] += (Math.random() - 0.5) * 0.15;
            // Keep particles in bounds
            if (Math.abs(positions[i]) > 2) positions[i] *= 0.5;
            if (Math.abs(positions[i + 1]) > 2) positions[i + 1] *= 0.5;
            if (Math.abs(positions[i + 2]) > 2) positions[i + 2] *= 0.5;
        }
        satellite.faultEffects.seu.geometry.attributes.position.needsUpdate = true;
    } else {
        seuMat.opacity = Math.max(0, seuMat.opacity - delta * 3);
    }
    
    // Thermal glow (orange/red pulsing)
    if (state.faults.thermal) {
        satellite.faultEffects.thermal.intensity = 4 + Math.sin(time / 100) * 1.5;
        satellite.faultEffects.thermal.color.setHSL(0.05 + Math.sin(time / 200) * 0.03, 1, 0.5);
    } else {
        satellite.faultEffects.thermal.intensity = Math.max(0, satellite.faultEffects.thermal.intensity - delta * 8);
    }
    
    // === POWER FAULT - EMERGENCY SHUTDOWN EFFECT ===
    if (state.faults.power) {
        // Intense flickering during shutdown, then darkness
        if (shutdownFactor > 0.2) {
            // Flickering phase
            const flicker = Math.sin(time / 20) * Math.sin(time / 37) * Math.sin(time / 53);
            satellite.faultEffects.power.intensity = 8 * Math.max(0, flicker) * shutdownFactor;
            satellite.faultEffects.power.color.setRGB(1, 0.8 * Math.random(), 0);
        } else {
            // Almost dead - occasional spark
            satellite.faultEffects.power.intensity = Math.random() > 0.95 ? 3 : 0;
        }
    } else {
        satellite.faultEffects.power.intensity = Math.max(0, satellite.faultEffects.power.intensity - delta * 8);
    }
    
    // Memory fault (purple glitch effect)
    if (state.faults.memory) {
        satellite.faultEffects.memory.intensity = 3 + Math.random() * 2;
        satellite.faultEffects.memory.color.setHSL(0.75 + Math.random() * 0.1, 1, 0.5);
    } else {
        satellite.faultEffects.memory.intensity = Math.max(0, satellite.faultEffects.memory.intensity - delta * 6);
    }
    
    // Latchup (intense red strobe)
    if (state.faults.latchup) {
        satellite.faultEffects.latchup.intensity = 6 * (Math.sin(time / 40) > 0 ? 1 : 0.2);
    } else {
        satellite.faultEffects.latchup.intensity = Math.max(0, satellite.faultEffects.latchup.intensity - delta * 10);
    }
    
    // Recovery glow fade (brighter and longer for power recovery)
    if (!state.faults.power && satellite.prevFaults.power) {
        // Just recovered from power fault - big recovery glow
        satellite.faultEffects.recoveryGlow.intensity = 10;
    }
    satellite.faultEffects.recoveryGlow.intensity = Math.max(0, satellite.faultEffects.recoveryGlow.intensity - delta * 3);
    
    // Satellite body dims during shutdown
    if (shutdownFactor < 1) {
        satellite.group.children.forEach(child => {
            if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshStandardMaterial) {
                // Dim all materials during power failure
                const mat = child.material;
                if (mat.emissive) {
                    // Apply shutdown dimming
                    mat.emissiveIntensity *= shutdownFactor;
                }
            }
        });
    }
}
