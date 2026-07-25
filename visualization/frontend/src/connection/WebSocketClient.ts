/**
 * WebSocket client for connecting to TITAN backend
 */

export class WebSocketClient {
    private ws: WebSocket | null = null;
    private url: string;
    private reconnectInterval: number = 5000;
    private messageHandler: ((data: any) => void) | null = null;
    private connectHandler: (() => void) | null = null;
    private disconnectHandler: (() => void) | null = null;
    private connected: boolean = false;

    constructor(url: string) {
        this.url = url;
        this.connect();
    }

    private connect(): void {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.connected = true;
                if (this.connectHandler) {
                    this.connectHandler();
                }
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (this.messageHandler) {
                        this.messageHandler(data);
                    }
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.connected = false;
                if (this.disconnectHandler) {
                    this.disconnectHandler();
                }
                // Attempt to reconnect
                setTimeout(() => this.connect(), this.reconnectInterval);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (e) {
            console.error('Failed to connect:', e);
            setTimeout(() => this.connect(), this.reconnectInterval);
        }
    }

    public send(data: object): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    public onMessage(handler: (data: any) => void): void {
        this.messageHandler = handler;
    }

    public onConnect(handler: () => void): void {
        this.connectHandler = handler;
    }

    public onDisconnect(handler: () => void): void {
        this.disconnectHandler = handler;
    }

    public isConnected(): boolean {
        return this.connected;
    }
}
