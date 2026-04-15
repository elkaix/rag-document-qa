import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";

export function useDocuments() {
  return useQuery({
    queryKey: ["documents"],
    queryFn: api.listDocuments,
    refetchInterval: 30_000,
  });
}

export function useDocumentChunks(docId: string | null) {
  return useQuery({
    queryKey: ["documents", docId, "chunks"],
    queryFn: () => api.getDocumentChunks(docId!),
    enabled: !!docId,
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => api.deleteDocument(docId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}
