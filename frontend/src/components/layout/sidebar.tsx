/**
 * Sidebar — conversation list, navigation, and settings.
 *
 * RAG Pipeline Position:
 *   This is a NAVIGATION component. It sits outside the pipeline but provides
 *   the primary interface for managing conversations (create, rename, pin,
 *   export, share, delete) and configuring retrieval settings (model, top-k).
 *
 * Layout (top to bottom):
 *   1. Logo/brand area with collapse toggle
 *   2. "+ New Chat" button
 *   3. Search input (debounced 300ms)
 *   4. Scrollable conversation list grouped by: Pinned, Today, Yesterday, This Week, Older
 *   5. Navigation links (Upload, Documents)
 *   6. Settings (Model dropdown, Top-K slider)
 *   7. Collection stats
 *
 * WHY: Conversations replace the old "Chat" nav link. Each conversation item
 *      is clickable to navigate to /chat/:id, with a context menu for actions.
 */

import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router";
import {
  Upload,
  FolderOpen,
  BarChart3,
  PanelLeftClose,
  PanelLeftOpen,
  FileText,
  Database,
  HardDrive,
  Layers,
  Plus,
  Search,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Download,
  Share2,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { useSettings, MODEL_OPTIONS } from "@/hooks/use-settings";
import { useDocuments } from "@/hooks/use-documents";
import { useConversations } from "@/hooks/use-conversations";
import { api } from "@/api/client";
import { cn } from "@/lib/utils";
import type { ConversationSummary } from "@/api/types";

// --- Navigation items (Chat is replaced by conversation list) ---

const NAV_ITEMS = [
  { to: "/upload", label: "Upload", icon: Upload },
  { to: "/documents", label: "Documents", icon: FolderOpen },
  { to: "/eval", label: "Evaluation", icon: BarChart3 },
] as const;

// --- Date grouping helper ---

/**
 * Groups conversations into time-based categories for the sidebar.
 *
 * WHY: Users expect recent conversations at the top. Grouping by relative
 *      time (Today, Yesterday, This Week, Older) matches the pattern from
 *      ChatGPT and similar UIs — familiar and scannable.
 *
 * PATTERN: Pinned conversations always appear first regardless of date.
 *          Within each group, conversations are sorted by updated_at descending
 *          (most recent first) since that's the API default.
 */
function groupByDate(conversations: ConversationSummary[]) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const weekAgo = new Date(today.getTime() - 7 * 86400000);

  const groups: { label: string; items: ConversationSummary[] }[] = [];
  const pinned = conversations.filter((c) => c.pinned);
  const unpinned = conversations.filter((c) => !c.pinned);

  if (pinned.length) groups.push({ label: "Pinned", items: pinned });

  const todayItems = unpinned.filter((c) => new Date(c.updated_at) >= today);
  const yesterdayItems = unpinned.filter((c) => {
    const d = new Date(c.updated_at);
    return d >= yesterday && d < today;
  });
  const weekItems = unpinned.filter((c) => {
    const d = new Date(c.updated_at);
    return d >= weekAgo && d < yesterday;
  });
  const olderItems = unpinned.filter(
    (c) => new Date(c.updated_at) < weekAgo,
  );

  if (todayItems.length) groups.push({ label: "Today", items: todayItems });
  if (yesterdayItems.length)
    groups.push({ label: "Yesterday", items: yesterdayItems });
  if (weekItems.length) groups.push({ label: "This Week", items: weekItems });
  if (olderItems.length) groups.push({ label: "Older", items: olderItems });

  return groups;
}

/**
 * Formats a date as a relative timestamp (e.g., "2h ago", "3d ago").
 *
 * WHY: Relative timestamps are more immediately useful than absolute dates
 *      in a sidebar where screen space is limited.
 */
function relativeTime(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return "now";
  if (diffMin < 60) return `${diffMin}m ago`;

  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;

  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;

  const diffMonth = Math.floor(diffDay / 30);
  return `${diffMonth}mo ago`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function Sidebar() {
  const [expanded, setExpanded] = useState(true);
  const { settings, update: updateSettings } = useSettings();
  const { data: documents } = useDocuments();
  const { conversations, create, remove, update } = useConversations();
  const navigate = useNavigate();
  const location = useLocation();

  // WHY: Debounced search — typing fires a search after 300ms of inactivity
  //      instead of on every keystroke. Reduces API calls and avoids UI jank.
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ConversationSummary | null>(null);
  const [searchResults, setSearchResults] = useState<
    ConversationSummary[] | null
  >(null);

  // BUG FIX: Previously called `setSearchResults(null)` synchronously when
  //          the query was empty, which triggers the react-hooks/set-state-
  //          in-effect cascade warning. Instead we derive "should we show
  //          search results?" at render time (see `displayConversations`
  //          below) and only run the fetch when the query is non-empty.
  useEffect(() => {
    if (!searchQuery.trim()) return;
    const timer = setTimeout(async () => {
      try {
        const results = await api.searchConversations(searchQuery);
        setSearchResults(results);
      } catch {
        setSearchResults([]);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // PATTERN: Derive active conversation ID from the current URL path.
  //          The sidebar renders at the layout level (parent of all routes),
  //          so we parse location.pathname rather than relying on useParams
  //          which only works inside the matched route.
  const activeConversationId = location.pathname.startsWith("/chat/")
    ? location.pathname.split("/chat/")[1]
    : undefined;

  // --- Conversation list data ---
  // WHY: Show search results only while the query is active. Clearing
  //      searchResults on empty-query would require a setState-in-effect,
  //      which lint (rightly) flags as a cascading render.
  const displayConversations =
    searchQuery.trim() && searchResults ? searchResults : conversations;
  const groups = groupByDate(displayConversations);

  // --- Collection stats ---
  const totalChunks =
    documents?.reduce((acc, doc) => acc + doc.chunks, 0) ?? 0;
  const totalSize =
    documents?.reduce((acc, doc) => acc + (doc.file_size_bytes ?? 0), 0) ?? 0;

  const modelLabel =
    MODEL_OPTIONS.find((m) => m.value === settings.model)?.label ??
    settings.model;

  // --- Action handlers ---

  async function handleNewChat() {
    const result = await create();
    navigate(`/chat/${result.id}`);
  }

  function handleRename(conv: ConversationSummary) {
    const newTitle = window.prompt("Rename conversation:", conv.title);
    if (newTitle && newTitle.trim() !== conv.title) {
      update({ id: conv.id, patch: { title: newTitle.trim() } });
    }
  }

  function handleTogglePin(conv: ConversationSummary) {
    update({ id: conv.id, patch: { pinned: !conv.pinned } });
  }

  async function handleExport(conv: ConversationSummary) {
    try {
      const text = await api.exportConversation(conv.id);
      // WHY: Create a temporary download link via Blob + URL.createObjectURL.
      //      This avoids needing a server-side file download endpoint.
      const blob = new Blob([text], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${conv.title.replace(/[^a-zA-Z0-9]/g, "_")}.md`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Conversation exported");
    } catch {
      toast.error("Failed to export conversation");
    }
  }

  async function handleShare(conv: ConversationSummary) {
    try {
      const { share_url } = await api.shareConversation(conv.id);
      await navigator.clipboard.writeText(share_url);
      toast.success("Share link copied to clipboard");
    } catch {
      toast.error("Failed to share conversation");
    }
  }

  function handleDelete(conv: ConversationSummary) {
    setDeleteTarget(conv);
  }

  function confirmDelete() {
    if (!deleteTarget) return;
    remove(deleteTarget.id);
    if (activeConversationId === deleteTarget.id) {
      navigate("/chat");
    }
    setDeleteTarget(null);
  }

  return (
    <aside
      data-slot="sidebar"
      className={cn(
        "flex h-screen flex-col bg-[#24292d] text-[#F3F4F6] transition-all duration-200",
        expanded ? "w-60" : "w-16",
      )}
    >
      {/* Hamburger toggle -- always visible at top */}
      <div
        className={cn(
          "flex h-[50px] items-center px-3 border-b border-[#3a4149]",
          expanded ? "justify-between" : "justify-center",
        )}
      >
        {expanded && (
          <span className="text-sm font-bold tracking-tight text-white">
            RAG <span className="text-[#2fbb4f]">Q&A</span>
          </span>
        )}
        <button
          className="rounded-md p-1.5 transition-colors"
          style={{ color: "#9CA3AF" }}
          onMouseEnter={(e) => { e.currentTarget.style.color = "#F3F4F6"; e.currentTarget.style.backgroundColor = "#3a3f44"; }}
          onMouseLeave={(e) => { e.currentTarget.style.color = "#9CA3AF"; e.currentTarget.style.backgroundColor = "transparent"; }}
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? (
            <PanelLeftClose className="size-4" />
          ) : (
            <PanelLeftOpen className="size-4" />
          )}
        </button>
      </div>

      {/* New Chat button */}
      {expanded ? (
        <div className="p-2">
          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start gap-2 border-[#3a4149] text-[#8b949e] hover:text-white hover:bg-[#2b3137]"
            onClick={handleNewChat}
          >
            <Plus className="size-4" />
            New Chat
          </Button>
        </div>
      ) : (
        <div className="flex justify-center p-2">
          <Tooltip>
            <TooltipTrigger
              render={
                <button
                  className="rounded-md p-1.5 transition-colors"
                  style={{ color: "#9CA3AF" }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = "#F3F4F6"; e.currentTarget.style.backgroundColor = "#3a3f44"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = "#9CA3AF"; e.currentTarget.style.backgroundColor = "transparent"; }}
                  onClick={handleNewChat}
                />
              }
            >
              <Plus className="size-4" />
            </TooltipTrigger>
            <TooltipContent side="right">New Chat</TooltipContent>
          </Tooltip>
        </div>
      )}

      {/* Search input -- only when expanded */}
      {expanded && (
        <div className="px-2 pb-2">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-[#8b949e]" />
            <Input
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-7 pl-7 text-xs placeholder:text-[#8b949e]"
              style={{ color: "#24292d", caretColor: "#24292d", backgroundColor: "#FFFFFF" }}
            />
          </div>
        </div>
      )}

      {/* Scrollable conversation list */}
      {expanded && (
        <ScrollArea className="flex-1 min-h-0">
          <div className="flex flex-col gap-1 p-2">
            {groups.length === 0 && (
              <p className="text-xs text-[#8b949e] text-center py-4">
                {searchQuery ? "No results found" : "No conversations yet"}
              </p>
            )}
            {groups.map((group) => (
              <div key={group.label} className="mb-2">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[#8b949e] px-2">
                  {group.label}
                </span>
                <div className="flex flex-col gap-0.5 mt-1">
                  {group.items.map((conv) => (
                    <ConversationItem
                      key={conv.id}
                      conversation={conv}
                      isActive={activeConversationId === conv.id}
                      onNavigate={() => navigate(`/chat/${conv.id}`)}
                      onRename={() => handleRename(conv)}
                      onTogglePin={() => handleTogglePin(conv)}
                      onExport={() => handleExport(conv)}
                      onShare={() => handleShare(conv)}
                      onDelete={() => handleDelete(conv)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}

      {/* Collapsed state: just show a chat icon */}
      {!expanded && (
        <div className="flex-1" />
      )}

      {/* Divider line */}
      <div className="mx-3 my-1 h-px bg-[#3a4149]" />

      {/* Navigation links (Upload, Documents) at bottom */}
      <nav className="flex flex-col gap-1 p-2">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
          const link = (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                  isActive
                    ? "bg-[#0d74e7]/15 text-[#5ba3f0] border-l-2 border-[#0d74e7]"
                    : "text-[#8b949e] hover:text-white hover:bg-[#2b3137] border-l-2 border-transparent",
                )
              }
            >
              <Icon className="size-5 shrink-0" />
              {expanded && <span>{label}</span>}
            </NavLink>
          );

          if (!expanded) {
            return (
              <Tooltip key={to}>
                <TooltipTrigger render={link} />
                <TooltipContent side="right">{label}</TooltipContent>
              </Tooltip>
            );
          }

          return link;
        })}
      </nav>

      {/* Divider line */}
      <div className="mx-3 my-1 h-px bg-[#3a4149]" />

      {/* Settings section -- only when expanded */}
      {expanded && (
        <div className="flex flex-col gap-4 p-3">
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Model
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full justify-start whitespace-nowrap overflow-hidden"
                  />
                }
              >
                {modelLabel}
              </DropdownMenuTrigger>
              <DropdownMenuContent className="min-w-[14rem]">
                <DropdownMenuRadioGroup
                  value={settings.model}
                  onValueChange={(value) =>
                    updateSettings({ model: value as string })
                  }
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <DropdownMenuRadioItem
                      key={opt.value}
                      value={opt.value}
                      className="whitespace-nowrap"
                    >
                      {/* PATTERN: flex row with label left, hint right-aligned.
                          Keeps the radio checkmark on the same row as the name
                          regardless of hint length. */}
                      <span className="flex w-full items-baseline justify-between gap-3">
                        <span>{opt.label}</span>
                        {opt.hint && (
                          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                            {opt.hint}
                          </span>
                        )}
                      </span>
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Top-K
              </span>
              <span className="text-xs tabular-nums text-muted-foreground">
                {settings.topK}
              </span>
            </div>
            <Slider
              min={1}
              max={20}
              value={[settings.topK]}
              onValueChange={(value) => {
                const v = Array.isArray(value) ? value[0] : value;
                updateSettings({ topK: v });
              }}
            />
          </div>
        </div>
      )}

      {/* Divider line */}
      <div className="mx-3 my-1 h-px bg-[#3a4149]" />

      {/* Collection stats -- only when expanded */}
      {expanded && documents && (
        <div className="grid grid-cols-2 gap-2 p-3 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <FileText className="size-3.5" />
            <span>{documents.length} docs</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Database className="size-3.5" />
            <span>{totalChunks} chunks</span>
          </div>
          <div className="flex items-center gap-1.5">
            <HardDrive className="size-3.5" />
            <span>{formatBytes(totalSize)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Layers className="size-3.5" />
            <span>
              {new Set(documents.map((d) => d.file_type).filter(Boolean)).size}{" "}
              types
            </span>
          </div>
        </div>
      )}
      <Dialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Conversation</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete{" "}
              <strong>{deleteTarget?.title}</strong>? This will remove all
              messages and sources. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  );
}

// --- Conversation list item with context menu ---

/**
 * Individual conversation item in the sidebar list.
 *
 * WHY: Extracted as a separate component to keep the sidebar readable and
 *      to isolate hover state management (the "..." menu button only appears
 *      on hover, matching the ChatGPT sidebar pattern).
 */
interface ConversationItemProps {
  conversation: ConversationSummary;
  isActive: boolean;
  onNavigate: () => void;
  onRename: () => void;
  onTogglePin: () => void;
  onExport: () => void;
  onShare: () => void;
  onDelete: () => void;
}

function ConversationItem({
  conversation,
  isActive,
  onNavigate,
  onRename,
  onTogglePin,
  onExport,
  onShare,
  onDelete,
}: ConversationItemProps) {
  return (
    <div
      className={cn(
        "group relative flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm cursor-pointer transition-all",
        isActive
          ? "bg-[#0d74e7]/15 text-[#5ba3f0]"
          : "text-[#8b949e] hover:text-white hover:bg-[#2b3137]",
      )}
      onClick={onNavigate}
    >
      {/* Conversation title and timestamp */}
      <div className="flex-1 min-w-0">
        <div className="truncate text-xs font-medium leading-snug">
          {conversation.title.length > 40
            ? conversation.title.slice(0, 40) + "..."
            : conversation.title}
        </div>
        <div className="text-[10px] text-[#8b949e] mt-0.5">
          {relativeTime(conversation.updated_at)}
        </div>
      </div>

      {/* Context menu — visible on hover */}
      <div
        className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
        onClick={(e) => e.stopPropagation()}
      >
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant="ghost"
                size="icon-xs"
                className="size-5 text-[#8b949e] hover:text-white"
              />
            }
          >
            <MoreHorizontal className="size-3" />
          </DropdownMenuTrigger>
          <DropdownMenuContent side="right" sideOffset={8}>
            <DropdownMenuItem onClick={onRename}>
              <Pencil className="size-3.5" />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onTogglePin}>
              {conversation.pinned ? (
                <>
                  <PinOff className="size-3.5" />
                  Unpin
                </>
              ) : (
                <>
                  <Pin className="size-3.5" />
                  Pin
                </>
              )}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onExport}>
              <Download className="size-3.5" />
              Export
            </DropdownMenuItem>
            <DropdownMenuItem onClick={onShare}>
              <Share2 className="size-3.5" />
              Share
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onClick={onDelete}>
              <Trash2 className="size-3.5" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
