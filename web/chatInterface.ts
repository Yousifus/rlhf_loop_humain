export class ChatInterface {
    private messages: { timestamp: string, sender: string, content: string }[] = [];
    private userName: string;

    constructor(userName: string) {
        this.userName = userName;
    }

    private formatMessage(message: string): string {
        // Replace **text** with <strong>text</strong>
        message = message.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Replace *text* with <em>text</em>
        message = message.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Replace `code` with <code>code</code>
        message = message.replace(/`(.*?)`/g, '<code>$1</code>');
        return message;
    }

    public sendMessage(message: string): void {
        const timestamp = new Date().toLocaleTimeString();
        this.messages.push({ timestamp, sender: this.userName, content: this.formatMessage(message) });
        console.log(`[${timestamp}] ${this.userName}: ${this.formatMessage(message)}`);
    }

    public receiveMessage(message: string): string {
        const timestamp = new Date().toLocaleTimeString();
        const formattedMessage = this.formatMessage(message);
        this.messages.push({ timestamp, sender: 'Assistant', content: formattedMessage });
        console.log(`[${timestamp}] Assistant: ${formattedMessage}`);
        return formattedMessage;
    }

    public getChatHistory(): string {
        return this.messages.map(msg => `[${msg.timestamp}] ${msg.sender}: ${msg.content}`).join('\n');
    }
}