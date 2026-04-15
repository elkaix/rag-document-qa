import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { UploadResponse } from "@/api/types";

export interface FileQueueItem {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  result?: UploadResponse;
  error?: string;
}

export function useUploadFile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => api.uploadFile(file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}
