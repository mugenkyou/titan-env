/**
 * Fault Recovery Chart - Shows how Causal RL resolves faults over time
 * Displays fault injections, recovery actions, and reward trends
 */

interface ChartDataPoint {
    step: number;
    faultCount: number;
    recoveryAction: boolean;
    reward: number;
    faultType?: string;
    actionName?: string;
    wasResolved?: boolean;  // Track if this step resolved a fault
}

const MAX_DATA_POINTS = 100;
const CHART_COLORS = {
    fault: '#ff4444',
    recovery: '#44ff88',
    reward: '#ffaa00',
    grid: 'rgba(255, 255, 255, 0.15)',
    axis: 'rgba(255, 255, 255, 0.4)',
    background: 'rgba(0, 20, 40, 0.6)',
    faultArea: 'rgba(255, 68, 68, 0.3)',
    resolvedBg: 'rgba(68, 255, 136, 0.15)',
};

class FaultRecoveryChart {
    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;
    private dataPoints: ChartDataPoint[] = [];
    private animationFrame: number = 0;
    
    constructor() {
        this.init();
    }
    
    private init(): void {
        this.canvas = document.getElementById('chart-canvas') as HTMLCanvasElement;
        if (!this.canvas) {
            console.warn('Chart canvas not found');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        if (!this.ctx) return;
        
        // Set canvas size
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Start animation loop
        this.animate();
    }
    
    private resize(): void {
        if (!this.canvas) return;
        const parent = this.canvas.parentElement;
        if (parent) {
            this.canvas.width = parent.clientWidth;
            this.canvas.height = parent.clientHeight;
        }
    }
    
    public addDataPoint(
        step: number,
        faultCount: number,
        recoveryAction: boolean,
        reward: number,
        faultType?: string,
        actionName?: string
    ): void {
        // Detect if a fault was just resolved (previous had faults, this recovery action reduced them)
        const prevPoint = this.dataPoints.length > 0 ? this.dataPoints[this.dataPoints.length - 1] : null;
        const wasResolved = prevPoint && prevPoint.faultCount > 0 && faultCount < prevPoint.faultCount;
        
        this.dataPoints.push({
            step,
            faultCount,
            recoveryAction,
            reward,
            faultType,
            actionName,
            wasResolved,
        });
        
        // Keep only recent data points
        if (this.dataPoints.length > MAX_DATA_POINTS) {
            this.dataPoints.shift();
        }
    }
    
    public reset(): void {
        this.dataPoints = [];
    }
    
    private animate(): void {
        this.draw();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    private draw(): void {
        if (!this.ctx || !this.canvas) return;
        
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        ctx.fillStyle = CHART_COLORS.background;
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid(ctx, width, height);
        
        if (this.dataPoints.length < 2) {
            // Draw "waiting for data" message
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '12px Consolas, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for simulation data...', width / 2, height / 2);
            return;
        }
        
        // Draw data
        const padding = { left: 5, right: 5, top: 20, bottom: 15 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        
        // Calculate scales
        const xScale = chartWidth / (this.dataPoints.length - 1);
        
        // Find max values for scaling
        const maxFaults = Math.max(5, ...this.dataPoints.map(d => d.faultCount));
        const maxReward = Math.max(1, ...this.dataPoints.map(d => Math.abs(d.reward)));
        
        // Draw resolution zones (green background where faults were resolved)
        this.drawResolutionZones(ctx, padding, chartHeight, xScale);
        
        // Draw fault count line (red area)
        this.drawFaultLine(ctx, padding, chartHeight, xScale, maxFaults);
        
        // Draw recovery action markers (green dots)
        this.drawRecoveryMarkers(ctx, padding, chartHeight, xScale);
        
        // Draw reward line (orange)
        this.drawRewardLine(ctx, padding, chartHeight, xScale, maxReward);
        
        // Draw fault injection markers (red triangles)
        this.drawFaultMarkers(ctx, padding, chartHeight, xScale);
        
        // Draw legend
        this.drawLegend(ctx, width);
    }
    
    private drawResolutionZones(
        ctx: CanvasRenderingContext2D,
        padding: { left: number; right: number; top: number; bottom: number },
        chartHeight: number,
        xScale: number
    ): void {
        // Highlight areas where faults were resolved with green background
        this.dataPoints.forEach((point, i) => {
            if (point.wasResolved) {
                const x = padding.left + i * xScale;
                ctx.fillStyle = CHART_COLORS.resolvedBg;
                ctx.fillRect(x - xScale/2, padding.top, xScale, chartHeight);
            }
        });
    }
    
    private drawLegend(ctx: CanvasRenderingContext2D, width: number): void {
        ctx.font = '10px Consolas, monospace';
        ctx.textAlign = 'left';
        
        // Fault indicator
        ctx.fillStyle = CHART_COLORS.fault;
        ctx.fillRect(5, 3, 10, 10);
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.fillText('Fault', 18, 11);
        
        // Recovery indicator  
        ctx.fillStyle = CHART_COLORS.recovery;
        ctx.beginPath();
        ctx.arc(70, 8, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.fillText('Recovery', 78, 11);
        
        // Reward indicator
        ctx.strokeStyle = CHART_COLORS.reward;
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(145, 8);
        ctx.lineTo(165, 8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        ctx.fillText('Reward', 168, 11);
    }
    
    private drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
        ctx.strokeStyle = CHART_COLORS.grid;
        ctx.lineWidth = 1;
        
        // Horizontal lines
        for (let i = 0; i < 5; i++) {
            const y = (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Vertical lines
        for (let i = 0; i < 10; i++) {
            const x = (width / 9) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }
    
    private drawFaultLine(
        ctx: CanvasRenderingContext2D,
        padding: { left: number; right: number; top: number; bottom: number },
        chartHeight: number,
        xScale: number,
        maxFaults: number
    ): void {
        if (this.dataPoints.length < 2) return;
        
        // Draw filled area
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top + chartHeight);
        
        this.dataPoints.forEach((point, i) => {
            const x = padding.left + i * xScale;
            const y = padding.top + chartHeight - (point.faultCount / maxFaults) * chartHeight;
            if (i === 0) {
                ctx.lineTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.lineTo(padding.left + (this.dataPoints.length - 1) * xScale, padding.top + chartHeight);
        ctx.closePath();
        
        // Fill with gradient
        const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartHeight);
        gradient.addColorStop(0, 'rgba(255, 68, 68, 0.4)');
        gradient.addColorStop(1, 'rgba(255, 68, 68, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // Draw line
        ctx.strokeStyle = CHART_COLORS.fault;
        ctx.lineWidth = 2;
        ctx.beginPath();
        this.dataPoints.forEach((point, i) => {
            const x = padding.left + i * xScale;
            const y = padding.top + chartHeight - (point.faultCount / maxFaults) * chartHeight;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
    }
    
    private drawRewardLine(
        ctx: CanvasRenderingContext2D,
        padding: { left: number; right: number; top: number; bottom: number },
        chartHeight: number,
        xScale: number,
        maxReward: number
    ): void {
        if (this.dataPoints.length < 2) return;
        
        const midY = padding.top + chartHeight / 2;
        
        ctx.strokeStyle = CHART_COLORS.reward;
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        
        this.dataPoints.forEach((point, i) => {
            const x = padding.left + i * xScale;
            // Normalize reward: positive goes up, negative goes down from middle
            const normalizedReward = point.reward / maxReward;
            const y = midY - normalizedReward * (chartHeight / 2) * 0.8;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        ctx.setLineDash([]);
    }
    
    private drawRecoveryMarkers(
        ctx: CanvasRenderingContext2D,
        padding: { left: number; right: number; top: number; bottom: number },
        chartHeight: number,
        xScale: number
    ): void {
        ctx.fillStyle = CHART_COLORS.recovery;
        ctx.shadowColor = CHART_COLORS.recovery;
        ctx.shadowBlur = 8;
        
        this.dataPoints.forEach((point, i) => {
            if (point.recoveryAction) {
                const x = padding.left + i * xScale;
                const y = padding.top + chartHeight - 15;
                
                // Draw glowing dot
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw upward arrow
                ctx.strokeStyle = CHART_COLORS.recovery;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y - 6);
                ctx.lineTo(x, y - 15);
                ctx.moveTo(x - 4, y - 11);
                ctx.lineTo(x, y - 15);
                ctx.lineTo(x + 4, y - 11);
                ctx.stroke();
            }
        });
        
        ctx.shadowBlur = 0;
    }
    
    private drawFaultMarkers(
        ctx: CanvasRenderingContext2D,
        padding: { left: number; right: number; top: number; bottom: number },
        chartHeight: number,
        xScale: number
    ): void {
        this.dataPoints.forEach((point, i) => {
            if (point.faultType) {
                const x = padding.left + i * xScale;
                const y = padding.top + 15;
                
                // Draw red triangle pointing down
                ctx.fillStyle = CHART_COLORS.fault;
                ctx.shadowColor = CHART_COLORS.fault;
                ctx.shadowBlur = 6;
                
                ctx.beginPath();
                ctx.moveTo(x, y + 8);
                ctx.lineTo(x - 6, y - 4);
                ctx.lineTo(x + 6, y - 4);
                ctx.closePath();
                ctx.fill();
                
                ctx.shadowBlur = 0;
            }
        });
    }
    
    public destroy(): void {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}

// Singleton instance
let chartInstance: FaultRecoveryChart | null = null;

export function initChart(): FaultRecoveryChart {
    if (!chartInstance) {
        chartInstance = new FaultRecoveryChart();
    }
    return chartInstance;
}

export function addChartData(
    step: number,
    faultCount: number,
    recoveryAction: boolean,
    reward: number,
    faultType?: string,
    actionName?: string
): void {
    if (chartInstance) {
        chartInstance.addDataPoint(step, faultCount, recoveryAction, reward, faultType, actionName);
    }
}

export function resetChart(): void {
    if (chartInstance) {
        chartInstance.reset();
    }
}
