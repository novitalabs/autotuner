# LLM Autotuner - Frontend

React + TypeScript frontend application for the LLM Autotuner.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **TanStack Query (React Query)** - Server state management
- **Axios** - HTTP client
- **Tailwind CSS** - Utility-first CSS (to be added)

## Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   │   └── Layout.tsx   # Main layout with navigation
│   ├── pages/           # Page components
│   │   ├── Dashboard.tsx
│   │   ├── Tasks.tsx
│   │   └── Experiments.tsx
│   ├── services/        # API client and services
│   │   └── api.ts       # API client with all endpoints
│   ├── types/           # TypeScript type definitions
│   │   └── api.ts       # API response/request types
│   ├── hooks/           # Custom React hooks (empty)
│   ├── utils/           # Utility functions (empty)
│   ├── App.tsx          # Main app component with routing
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── index.html           # HTML template
├── vite.config.ts       # Vite configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies and scripts
```

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

## Getting Started

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

3. **Build for production:**
   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

### API Configuration

The frontend communicates with the backend API. Configure the API URL in `.env`:

```bash
VITE_API_URL=http://localhost:8000/api
```

The development server includes a proxy configuration in `vite.config.ts` that forwards `/api` requests to the backend.

### Adding Features

This is a scaffold without concrete features. To add features:

1. **Create new pages** in `src/pages/`
2. **Add routes** in `src/App.tsx`
3. **Create components** in `src/components/`
4. **Add API calls** in `src/services/api.ts`
5. **Define types** in `src/types/`
6. **Use React Query** for data fetching with `useQuery` and `useMutation`

### Example: Fetching Tasks

```typescript
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';

function TaskList() {
  const { data: tasks, isLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: () => apiClient.getTasks(),
  });

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      {tasks?.map(task => (
        <div key={task.id}>{task.task_name}</div>
      ))}
    </div>
  );
}
```

## Features to Implement

The scaffold includes placeholder pages for:

- **Dashboard** - System overview and quick stats
- **Tasks** - Task management (create, list, start, monitor)
- **Experiments** - Experiment results and comparisons

## Notes

- The current setup uses basic CSS. Consider adding Tailwind CSS for better styling
- API types in `src/types/api.ts` match the backend schema
- React Query is configured for optimal performance
- All API endpoints are typed for TypeScript safety

## Troubleshooting

**Port already in use:**
```bash
# Change port in vite.config.ts server.port
```

**API connection errors:**
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings in backend
- Verify proxy configuration in `vite.config.ts`

**Type errors:**
```bash
npm run type-check
```
