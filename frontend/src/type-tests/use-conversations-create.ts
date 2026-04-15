import { useConversations } from "@/hooks/use-conversations";

// WHY: Starting a new chat from the sidebar should not require the caller to
//      provide a title. This type-level regression check keeps that API shape
//      valid during `tsc -b` runs without adding a runtime dependency.
type CreateConversationFn = ReturnType<typeof useConversations>["create"];

declare const createConversation: CreateConversationFn;

void createConversation();
