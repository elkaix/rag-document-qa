import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";

export function useConversations() {
  const qc = useQueryClient();

  const listQuery = useQuery({
    queryKey: ["conversations"],
    queryFn: api.listConversations,
    refetchInterval: 30_000,
  });

  const createMutation = useMutation({
    mutationFn: (title?: string) => api.createConversation(title),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: { title?: string; pinned?: boolean } }) =>
      api.updateConversation(id, patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["conversations"] }),
  });

  const searchFn = (q: string) => api.searchConversations(q);

  // WHY: React Query's mutateAsync type still expects a variables argument even
  //      when the mutation function makes it optional. Wrapping it preserves
  //      the UI contract: callers can create a conversation with the default
  //      title by calling `create()` with no arguments.
  const createConversation = (title?: string) => createMutation.mutateAsync(title);

  return {
    conversations: listQuery.data ?? [],
    isLoading: listQuery.isLoading,
    create: createConversation,
    remove: deleteMutation.mutate,
    update: updateMutation.mutate,
    search: searchFn,
  };
}
