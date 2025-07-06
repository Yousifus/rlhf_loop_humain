import { ChatInterface } from './chatInterface';
import * as readline from 'readline';
import WebSocket from 'ws';

const chat = new ChatInterface('User');

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws: WebSocket) => {
    console.log('New client connected');

    ws.on('message', (message: string) => {
        console.log(`Received: ${message}`);
        handleUserMessage(ws, message);
    });
    
    function handleUserMessage(ws: WebSocket, message: string) {
        const botResponse = chat.receiveMessage(message);
        ws.send(botResponse);
    }

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

// Commenting out readline interface for WebSocket communication
// const rl = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout
// });

// Commenting out promptUser function for WebSocket communication
// function promptUser() {
//     rl.question('You: ', (message: string) => {
//         if (message.toLowerCase() === 'exit') {
//             rl.close();
//             console.log("\nChat History:");
//             console.log(chat.getChatHistory());
//             return;
//         }
//         chat.sendMessage(message);
//         chat.receiveMessage('Your message has been received. How can I assist you further?');
//         promptUser();
//     });
// }

console.log('WebSocket server is running on ws://localhost:8080');