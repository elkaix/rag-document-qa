import { useState } from "react";
import { useDocumentChunks } from "@/hooks/use-documents";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";

interface ChunkViewerProps {
  docId: string;
}

export function ChunkViewer({ docId }: ChunkViewerProps) {
  const { data, isLoading } = useDocumentChunks(docId);
  const [filter, setFilter] = useState("");

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-4">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    );
  }

  const chunks = data?.chunks ?? [];
  const filtered = filter
    ? chunks.filter((c) =>
        c.excerpt.toLowerCase().includes(filter.toLowerCase())
      )
    : chunks;

  return (
    <div className="flex flex-col gap-3 p-4">
      <Input
        placeholder="Filter chunks..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="max-w-xs"
      />
      <ScrollArea className="max-h-64">
        <div className="flex flex-col gap-2">
          {filtered.map((chunk) => (
            <div
              key={chunk.chunk_id}
              className="rounded-lg p-3 text-xs leading-relaxed text-muted-foreground bg-[#F9FAFB] border-l-2 border-[#0d74e7]/30"
            >
              {chunk.excerpt}
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="py-4 text-center text-xs text-muted-foreground">
              No chunks found.
            </p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
