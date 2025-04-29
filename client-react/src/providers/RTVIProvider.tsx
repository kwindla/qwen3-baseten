'use client';

import { RTVIClient } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import { RTVIClientProvider } from '@pipecat-ai/client-react';
import { PropsWithChildren, useEffect, useState } from 'react';

export function RTVIProvider({ children }: PropsWithChildren) {
  const [client, setClient] = useState<RTVIClient | null>(null);

  useEffect(() => {
    console.log('Setting up Transport and Client');
    const transport = new DailyTransport();

    const rtviClient = new RTVIClient({
      transport,
      params: {
        baseUrl: '/api',
        endpoints: {
          connect: '/connect',
        },
      },
      enableCam: false,
    });

    setClient(rtviClient);
  }, []);

  if (!client) {
    return null;
  }

  return <RTVIClientProvider client={client}>{children}</RTVIClientProvider>;
}
