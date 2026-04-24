import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { Dropzone } from "@/components/upload/dropzone";
import { FileQueue } from "@/components/upload/file-queue";
import { useUploadFile } from "@/hooks/use-upload";
import type { FileQueueItem } from "@/hooks/use-upload";

export default function UploadPage() {
  const [queue, setQueue] = useState<FileQueueItem[]>([]);
  const uploadMutation = useUploadFile();
  // PATTERN: mirror queue into a ref so handleFiles can read the *current*
  //          length synchronously (state reads via closure would be stale
  //          across back-to-back drops). Writing the ref must happen in an
  //          effect — direct assignment during render is flagged by lint.
  const queueRef = useRef<FileQueueItem[]>([]);
  useEffect(() => {
    queueRef.current = queue;
  }, [queue]);

  const uploadFile = useCallback(
    async (file: File, startIndex: number) => {
      setQueue((prev) =>
        prev.map((q, idx) => (idx === startIndex ? { ...q, status: "uploading" } : q))
      );

      try {
        const result = await uploadMutation.mutateAsync(file);
        setQueue((prev) =>
          prev.map((q, idx) =>
            idx === startIndex ? { ...q, status: "done", result } : q
          )
        );
        toast.success(`${file.name} indexed (${result.chunks_count} chunks)`);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Upload failed";
        setQueue((prev) =>
          prev.map((q, idx) =>
            idx === startIndex ? { ...q, status: "error", error: message } : q
          )
        );
        toast.error(`Failed to upload ${file.name}: ${message}`);
      }
    },
    [uploadMutation]
  );

  const handleFiles = useCallback(
    (files: File[]) => {
      const baseIndex = queueRef.current.length;
      const newItems: FileQueueItem[] = files.map((file) => ({
        file,
        status: "uploading" as const,
      }));
      setQueue((prev) => [...prev, ...newItems]);

      files.forEach((file, i) => {
        uploadFile(file, baseIndex + i);
      });
    },
    [uploadFile]
  );

  return (
    <div className="h-full overflow-auto flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Upload Documents</h1>
        <p className="text-sm text-muted-foreground">
          Drop or select files to index them for question answering.
        </p>
      </div>

      <Dropzone onFiles={handleFiles} />

      {queue.length > 0 && (
        <div className="flex flex-col gap-4">
          <p className="text-sm text-muted-foreground">
            {queue.length} file{queue.length !== 1 ? "s" : ""}
          </p>
          <FileQueue items={queue} />
        </div>
      )}
    </div>
  );
}
