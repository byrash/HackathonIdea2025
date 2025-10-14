# Fraudulent Bank Check Detection - Frontend

Angular + Vite frontend for check fraud detection.

## Prerequisites

- Node.js 18 or higher
- npm or yarn

## Setup

1. **Install dependencies:**

```bash
cd frontend
npm install
```

## Running the Development Server

```bash
npm run serve
```

The application will be available at: `http://localhost:4200`

## Build for Production

```bash
npm run build
```

Build output will be in `dist/` directory.

## Features

### Upload Interface

- Drag-and-drop file upload
- Support for JPEG, PNG, PDF
- File validation (type and size)

### Real-Time Progress

- Live progress bar (0-100%)
- Stage-by-stage updates
- Visual stage indicators
- Status messages via Server-Sent Events (SSE)

### Results Dashboard

- Fraud risk score visualization
- Verdict display (LEGITIMATE/SUSPICIOUS/FRAUDULENT)
- Extracted check data
- Forensic analysis details
- Security features assessment
- Image comparison (original vs annotated)
- Processing timeline

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── upload.component.ts      # Upload UI
│   │   │   ├── progress.component.ts    # Progress display
│   │   │   └── results.component.ts     # Results view
│   │   ├── services/
│   │   │   └── api.service.ts           # API + SSE
│   │   └── app.component.ts             # Main app
│   ├── main.ts                          # Bootstrap
│   └── styles.css                       # Global styles
├── index.html
├── vite.config.ts
├── tailwind.config.js
└── package.json
```

## Technology Stack

- **Framework:** Angular 18 (standalone components)
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **HTTP:** Angular HttpClient
- **SSE:** Native EventSource API

## Connecting to Backend

By default, the frontend connects to `http://localhost:8000`.

To change the API URL, edit `src/app/services/api.service.ts`:

```typescript
private apiUrl = 'http://localhost:8000/api';
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

Server-Sent Events (SSE) are supported in all modern browsers.
