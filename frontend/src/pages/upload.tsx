import { useState, useCallback } from "react";
import { Upload } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dropzone } from "@/components/upload/dropzone";
import { FileQueue } from "@/components/upload/file-queue";
import { useUploadFile } from "@/hooks/use-upload";
import type { FileQueueItem } from "@/hooks/use-upload";

export default function UploadPage() {
  const [queue, setQueue] = useState<FileQueueItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const uploadMutation = useUploadFile();

  const handleFiles = useCallback((files: File[]) => {
    const newItems: FileQueueItem[] = files.map((file) => ({
      file,
      status: "pending",
    }));
    setQueue((prev) => [...prev, ...newItems]);
  }, []);

  async function handleUploadAll() {
    setUploading(true);

    for (let i = 0; i < queue.length; i++) {
      const item = queue[i];
      if (item.status !== "pending") continue;

      // Mark uploading
      setQueue((prev) =>
        prev.map((q, idx) => (idx === i ? { ...q, status: "uploading" } : q))
      );

      try {
        const result = await uploadMutation.mutateAsync(item.file);
        setQueue((prev) =>
          prev.map((q, idx) =>
            idx === i ? { ...q, status: "done", result } : q
          )
        );
        toast.success(`${item.file.name} indexed (${result.chunks_count} chunks)`);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Upload failed";
        setQueue((prev) =>
          prev.map((q, idx) =>
            idx === i ? { ...q, status: "error", error: message } : q
          )
        );
        toast.error(`Failed to upload ${item.file.name}: ${message}`);
      }
    }

    setUploading(false);
  }

  const pendingCount = queue.filter((q) => q.status === "pending").length;

  return (
    <div className="h-full overflow-auto flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Upload Documents</h1>
        <p className="text-sm text-muted-foreground">
          Upload files to index them for question answering.
        </p>
      </div>

      <Dropzone onFiles={handleFiles} />

      {queue.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {queue.length} file{queue.length !== 1 ? "s" : ""} in queue
            </p>
            {pendingCount > 0 && (
              <Button onClick={handleUploadAll} disabled={uploading}>
                <Upload className="mr-1.5 size-4" />
                Upload All ({pendingCount})
              </Button>
            )}
          </div>
          <FileQueue items={queue} />
        </div>
      )}
    </div>
  );
}
