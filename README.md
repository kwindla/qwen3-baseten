# Qwen 3 Chatbot: Baseten inference, Pipecat Cloud, Vercel

Simple chatbot with an SGLang/Baseten processor in `server/baseten_llm_hack.py`. Also includes a Next.js client for interacting with a bot server through Daily.co's WebRTC transport. Deployable to Pipecat Cloud and Vercel.

<img src="image.png" width="420px">

## Project Overview

- **Server**: Python-based Pipecat bot with video/audio processing capabilities
- **Client**: Next.js TypeScript web application using the Pipecat React & JS SDKs
- **Infrastructure**: Deployable to Pipecat Cloud (server) and Vercel (client)

## Quick Start

### 1. Server Setup

Navigate to the server directory:

```bash
cd server
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Copy env.example to .env and add your API keys:

```bash
cp env.example .env
# Edit .env to add OPENAI_API_KEY and CARTESIA_API_KEY
```

Run the server locally to test before deploying:

```bash
LOCAL_RUN=1 python bot.py
```

This will open a browser window with a Daily.co room where you can test your bot directly.

### 2. Client Setup

In a separate terminal, navigate to the client directory:

```bash
cd client-react
```

Install dependencies:

```bash
npm install
```

Create `.env.local` file with your Pipecat Cloud API key:

```bash
cp env.local.example .env.local
```

> Create a Pipecat Cloud API key using the dashboard

Start the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to interact with your agent through the Next.js client.

## Deployment

> See the [Pipecat Cloud Quickstart](https://docs.pipecat.daily.co/quickstart) for a complete walkthrough.

### Deploy Server to Pipecat Cloud

1. Install the Pipecat Cloud CLI:

```bash
pip install pipecatcloud
```

2. Authenticate:

```bash
pcc auth login
```

3. Build and push your Docker image:

```bash
cd server
chmod +x build.sh
./build.sh
```

> IMPORTANT: Before running this build script, you need to add your DOCKER_USERNAME

4. Create a secret set for your API keys:

```bash
pcc secrets set simple-chatbot-secrets --file .env
```

5. Deploy to Pipecat Cloud:

```bash
pcc deploy
```

> IMPORTANT: Before deploying, you need to add your Docker Hub username

### Deploy Client to Vercel

1. Push your Next.js client to GitHub

2. Connect your GitHub repository to Vercel

3. Add your `PIPECAT_CLOUD_API_KEY` environment variable in Vercel

4. Deploy with the Vercel dashboard or CLI
