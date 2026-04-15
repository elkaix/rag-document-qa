# React Frontend for RAG Document Q&A

**Date:** 2026-04-14
**Status:** Approved
**Replaces:** `streamlit_app/` (to be removed after React frontend is complete)

## Goal

Replace the Streamlit frontend with a React SPA that communicates with the existing FastAPI backend at `:8001`. The React app will be a production-grade, visually distinctive portfolio piece.

## Architecture

```
frontend/                  ← New React SPA (Vite)
  src/
    api/
      client.ts            ← Fetch wrapper with base URL, error handling
      types.ts             ← TypeScript types matching API response shapes
    hooks/
      use-chat.ts          ← WebSocket streaming + chat state management
      use-documents.ts     ← TanStack Query hooks for documents CRUD
      use-upload.ts        ← File upload mutation with progress tracking
    components/
      layout/
        sidebar.tsx        ← Collapsible nav sidebar with settings + stats
        page-header.tsx    ← Page title + breadcrumb
      chat/
        chat-thread.tsx    ← Scrollable message list
        chat-input.tsx     ← Text input with send button
        chat-message.tsx   ← Single message bubble (user vs assistant)
        sources-panel.tsx  ← Right-side panel with source citations
      upload/
        dropzone.tsx       ← Drag-and-drop file upload area
        file-list.tsx      ← Pending files table
        upload-progress.tsx ← Per-file progress indicators
      documents/
        doc-table.tsx      ← Document list with actions
        doc-stats.tsx      ← Stats cards (docs, chunks, size, types)
        chunk-viewer.tsx   ← Expandable chunk list per document
      ui/                  ← shadcn/ui components (auto-generated)
    pages/
      chat.tsx             ← /chat (default route)
      upload.tsx           ← /upload
      documents.tsx        ← /documents
    lib/
      utils.ts             ← cn() helper, formatters
    App.tsx                ← Router + QueryClientProvider + layout shell
    main.tsx               ← React DOM entry point
  index.html
  vite.config.ts
  tailwind.config.ts       ← (managed by shadcn init)
  tsconfig.json
  package.json

src/                       ← Existing Python backend (unchanged)
src/api/                   ← Existing FastAPI (unchanged)
```

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Build | Vite | 8.x |
| Framework | React + TypeScript | 19.2.x |
| Routing | react-router | 7.9.x |
| Server state | @tanstack/react-query | 5.90.x |
| UI components | shadcn/ui | 3.5.x (Tailwind v4) |
| Icons | lucide-react | latest |
| WebSocket | Native browser API | N/A |

## API Surface (existing, no changes needed)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/upload` | Upload single file |
| POST | `/api/upload/batch` | Upload multiple files |
| POST | `/api/query` | One-shot RAG query |
| WS | `/api/chat` | Streaming chat (token-by-token) |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/{doc_id}` | Delete document + chunks |
| GET | `/api/documents/{doc_id}/chunks` | List chunks for a document |
| GET | `/health` | Health check |

## Pages

### 1. Chat (`/chat` — default route)

Split layout: 60% chat thread on the left, 40% sources panel on the right.

**Left panel (chat thread):**
- Scrollable message list with auto-scroll on new messages
- User messages right-aligned, assistant messages left-aligned
- Streaming: tokens appear one by one via WebSocket (`/api/chat`)
- Text input at the bottom with send button (Enter to send, Shift+Enter for newline)
- "Clear chat" button in the header

**Right panel (sources):**
- Updates after each assistant response completes (from the WebSocket `done` event)
- Each source shows: filename, chunk index, relevance score with color-coded bar, excerpt
- Collapsible source cards, first one expanded by default
- "No sources yet" empty state when no queries have been made

**WebSocket protocol:**
```
→ Send:    {"query": "...", "top_k": 5, "model": "glm-5.1"}
← Receive: {"type": "token", "content": "..."}  (repeated)
← Receive: {"type": "done", "sources": [...]}    (final)
← Receive: {"type": "error", "content": "..."}   (on failure)
```

### 2. Upload (`/upload`)

- Drag-and-drop zone (accepts PDF, TXT, MD, HTML, CSV, DOCX, JSON)
- File queue table showing name, size, type, status (pending/uploading/done/error)
- "Upload All" button triggers sequential uploads via `POST /api/upload`
- Per-file progress indicator
- On success: invalidate documents query cache (auto-refreshes stats everywhere)
- Already-indexed detection: compare filename against documents list

### 3. Documents (`/documents`)

- Stats cards row: total docs, total chunks, total size, file types count
- Document table with columns: icon, filename, type, size, chunks, uploaded date, delete button
- Search/filter input for filename
- Expandable row to view chunks (lazy-loaded via `GET /api/documents/{doc_id}/chunks`)
- Chunk text with search/highlight within chunks
- "Delete All" with confirmation dialog
- Delete single document with confirmation

## Layout Shell

**Sidebar (always visible):**
- Narrow by default (~64px, icon-only), expands to ~240px on hover or toggle
- Navigation: 3 page links (Chat, Upload, Documents) with active state indicator
- Settings section (in a popover or inline when expanded):
  - Model selector dropdown (GLM 5.1, GPT-4, GPT-3.5, Llama 3)
  - Top-K slider (1-20)
  - Chunk size display (read-only, set at backend init)
- Collection stats at bottom: docs count, chunks count, index size
- Stats auto-refresh via TanStack Query (revalidate after upload/delete)

**Main area:**
- Takes remaining viewport width
- Each page renders here via react-router `<Outlet />`

## Data Flow

### Server State (TanStack Query)
- `["documents"]` — `GET /api/documents` — document list + metadata
- `["documents", docId, "chunks"]` — `GET /api/documents/{docId}/chunks` — lazy per-document
- `["health"]` — `GET /health` — optional health indicator

### Mutations
- `uploadFile` — `POST /api/upload` — invalidates `["documents"]` on success
- `deleteDocument` — `DELETE /api/documents/{doc_id}` — invalidates `["documents"]`

### Local State
- Chat history (array of messages) — not persisted, lives in `useChat` hook
- Current sources (from latest WebSocket `done` event)
- Sidebar collapsed/expanded state
- Settings (model, top_k) — stored in localStorage for persistence

## API Client (`api/client.ts`)

Thin wrapper around `fetch`:
- Base URL from `VITE_API_URL` env var (default: `http://localhost:8001`)
- Automatic JSON parsing
- Error handling (throws typed errors)
- No auth needed (open API)

## Custom Hooks

### `useChat`
- Manages WebSocket connection to `/api/chat`
- Exposes: `messages`, `sources`, `isStreaming`, `sendMessage(query)`, `clearChat()`
- Reads `model` and `topK` from settings context/localStorage
- Handles reconnection on disconnect

### `useDocuments`
- `useDocuments()` — wraps `useQuery(["documents"])` 
- `useDeleteDocument()` — wraps `useMutation` + invalidation
- `useDocumentChunks(docId)` — wraps `useQuery(["documents", docId, "chunks"], { enabled: !!docId })`

### `useUpload`
- `useUploadFile()` — wraps `useMutation` for single file upload
- Tracks per-file upload state (pending, uploading, done, error)
- Invalidates `["documents"]` on success

## Design Direction

Fresh, distinctive design (not the existing dark+amber theme). Specific aesthetic to be determined during implementation using the `frontend-design` skill. Key requirements:
- High contrast, readable typography
- Distinctive — not generic shadcn defaults
- Professional portfolio quality
- Responsive (works on desktop, graceful on tablet)

## shadcn/ui Components Needed

From the shadcn registry (installed via `npx shadcn@latest add`):
- `button`, `input`, `textarea`
- `card`, `badge`, `separator`
- `dialog` (delete confirmations)
- `dropdown-menu` (model selector)
- `slider` (top-k)
- `table` (documents list)
- `collapsible` (chunk viewer)
- `tooltip` (sidebar icons)
- `scroll-area` (chat thread)
- `progress` (upload progress)
- `skeleton` (loading states)
- `sonner` (toast notifications)

## Docker Integration

Update `docker-compose.yml` to add a frontend service:
```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
  ports:
    - "3000:3000"
  environment:
    - VITE_API_URL=http://api:8001
  depends_on:
    - api
```

Frontend Dockerfile: multi-stage build (node for build, nginx for serve).

## Migration Plan

1. Build the React frontend in `frontend/` — fully independent, no changes to backend
2. Verify all features work end-to-end
3. Remove `streamlit_app/` directory
4. Update root `docker-compose.yml`
5. Update `README.md`

## Out of Scope

- Authentication / user accounts
- Persistent chat history (backend storage)
- Mobile-first responsive design (desktop-first, tablet-acceptable)
- Server-side rendering
- Internationalization
