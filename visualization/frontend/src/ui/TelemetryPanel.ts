/**
 * Telemetry UI updates
 * Based on actual dataset columns from telemetry_causal_rl.csv:
 * - obs_voltage, obs_current_draw, obs_battery_soc
 * - obs_cpu_temperature, obs_power_temperature
 * - obs_memory_integrity, obs_cpu_load
 */

import { SatelliteState } from '../satellite/Satellite';

// Only include telemetry values that exist in the actual dataset
const TELEMETRY_CONFIG = [
    { key: 'battery_soc', label: 'Battery SOC', unit: '%', mult: 100, warn: 0.3, danger: 0.15, icon: '🔋' },
    { key: 'voltage', label: 'Voltage', unit: 'V', mult: 28, warn: 0.7, danger: 0.5, icon: '⚡' },
    { key: 'current_draw', label: 'Current Draw', unit: 'A', mult: 10, warn: 0.8, danger: 0.95, icon: '📊' },
    { key: 'cpu_temperature', label: 'CPU Temp', unit: '°C', mult: 100, warn: 0.7, danger: 0.85, icon: '🌡️' },
    { key: 'power_temperature', label: 'Power Temp', unit: '°C', mult: 100, warn: 0.7, danger: 0.85, icon: '🔥' },
    { key: 'cpu_load', label: 'CPU Load', unit: '%', mult: 100, warn: 0.8, danger: 0.95, icon: '💻' },
    { key: 'memory_integrity', label: 'Memory Integrity', unit: '%', mult: 100, warn: 0.5, danger: 0.15, icon: '💾' },
];

let initialized = false;

function initTelemetryRows() {
    const container = document.getElementById('telemetry-rows');
    if (!container || initialized) return;
    
    TELEMETRY_CONFIG.forEach(config => {
        const row = document.createElement('div');
        row.className = 'telemetry-row';
        row.innerHTML = `
            <span class="telemetry-label">${config.icon} ${config.label}</span>
            <span class="telemetry-value" id="val-${config.key}">--</span>
            <div class="telemetry-bar">
                <div class="telemetry-bar-fill" id="bar-${config.key}"></div>
            </div>
        `;
        container.appendChild(row);
    });
    
    initialized = true;
}

export function updateTelemetryUI(telemetry: Record<string, number>): void {
    initTelemetryRows();
    
    TELEMETRY_CONFIG.forEach(config => {
        const value = telemetry[config.key] ?? 0;
        const displayValue = (value * config.mult).toFixed(1);
        
        const valEl = document.getElementById(`val-${config.key}`);
        const barEl = document.getElementById(`bar-${config.key}`);
        
        if (valEl) {
            valEl.textContent = `${displayValue}${config.unit}`;
        }
        
        if (barEl) {
            barEl.style.width = `${Math.min(value * 100, 100)}%`;
            barEl.className = 'telemetry-bar-fill';
            
            // Apply warning/danger classes based on thresholds
            if (config.danger !== undefined && config.warn !== undefined) {
                // Check if this is a "lower is worse" metric
                const lowerIsWorse = config.key.includes('integrity') || 
                                     config.key.includes('battery') || 
                                     config.key.includes('voltage');
                
                if (lowerIsWorse) {
                    // Lower value = more danger (battery, memory, voltage)
                    if (value < config.danger) barEl.classList.add('danger');
                    else if (value < config.warn) barEl.classList.add('warning');
                } else {
                    // Higher value = more danger (temp, load, current)
                    if (value > config.danger) barEl.classList.add('danger');
                    else if (value > config.warn) barEl.classList.add('warning');
                }
            }
        }
    });
}

export function updateFaultUI(faults: Record<string, boolean>): void {
    const faultMap: Record<string, string> = {
        seu: 'fault-seu',
        latchup: 'fault-latchup',
        thermal: 'fault-thermal',
        memory: 'fault-memory',
        power: 'fault-power',
    };
    
    Object.entries(faultMap).forEach(([key, id]) => {
        const el = document.getElementById(id);
        if (el) {
            if (faults[key]) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        }
    });
}

const ACTION_NAMES: Record<number, string> = {
    [-1]: '⚡ FAULT INJECTED',
    0: 'NO_ACTION',
    1: '🔧 SUBSYSTEM_RESET',
    2: '💾 MEMORY_SCRUB',
    3: '⚡ LOAD_SHEDDING',
    4: '🔄 POWER_CYCLE',
    5: '❄️ THERMAL_THROTTLING',
    6: '🔒 ISOLATE_SUBSYSTEM',
};

let actionLogEntries: HTMLElement[] = [];
const MAX_LOG_ENTRIES = 15;

export function addActionLog(step: number, action: string | number, reason: string): void {
    const logContainer = document.getElementById('action-log');
    if (!logContainer) return;
    
    // Get action name
    let actionName: string;
    if (typeof action === 'number') {
        actionName = ACTION_NAMES[action] || `ACTION_${action}`;
    } else {
        actionName = action;
    }
    
    // Skip NO_ACTION entries to reduce clutter (unless reason is interesting)
    if (actionName === 'NO_ACTION' && reason === 'heuristic') {
        return;  // Skip boring entries
    }
    
    // Determine color based on action type
    let entryClass = 'action-entry';
    if (action === -1 || actionName.includes('INJECTED')) {
        entryClass += ' fault-inject';
    } else if (action !== 0 && action !== 'NO_ACTION') {
        entryClass += ' recovery-action';
    }
    
    const entry = document.createElement('div');
    entry.className = entryClass;
    entry.innerHTML = `
        <span class="step">[${step}]</span>
        <span class="action">${actionName}</span>
        <span class="reason">${reason}</span>
    `;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    actionLogEntries.unshift(entry);
    
    // Remove old entries
    while (actionLogEntries.length > MAX_LOG_ENTRIES) {
        const old = actionLogEntries.pop();
        if (old) logContainer.removeChild(old);
    }
}

export function updateStatsUI(state: SatelliteState): void {
    const stepEl = document.getElementById('stat-step');
    const episodeEl = document.getElementById('stat-episode');
    const rewardEl = document.getElementById('stat-reward');
    const recoveredEl = document.getElementById('stat-recovered');
    
    if (stepEl) stepEl.textContent = state.step.toString();
    if (episodeEl) episodeEl.textContent = state.episode.toString();
    if (rewardEl) rewardEl.textContent = state.totalReward.toFixed(2);
    if (recoveredEl) recoveredEl.textContent = state.faultsRecovered.toString();
}
