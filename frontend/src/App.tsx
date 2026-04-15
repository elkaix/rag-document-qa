import { createBrowserRouter, Navigate, RouterProvider } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { AppLayout } from "@/components/layout/app-layout";
import ChatPage from "@/pages/chat";
import UploadPage from "@/pages/upload";
import DocumentsPage from "@/pages/documents";
import SharedPage from "@/pages/shared";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10_000, retry: 1 },
  },
});

const router = createBrowserRouter([
  {
    Component: AppLayout,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: "chat/:conversationId?", Component: ChatPage },
      { path: "upload", Component: UploadPage },
      { path: "documents", Component: DocumentsPage },
    ],
  },
  { path: "shared/:token", Component: SharedPage },
]);

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delay={300}>
        <RouterProvider router={router} />
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
