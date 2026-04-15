import { useState } from "react";
import { NavLink } from "react-router";
import {
  MessageSquare,
  Upload,
  FolderOpen,
  Menu,
  X,
  FileText,
  Database,
  HardDrive,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { useSettings, MODEL_OPTIONS } from "@/hooks/use-settings";
import { useDocuments } from "@/hooks/use-documents";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/upload", label: "Upload", icon: Upload },
  { to: "/documents", label: "Documents", icon: FolderOpen },
] as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function Sidebar() {
  const [expanded, setExpanded] = useState(true);
  const { settings, update } = useSettings();
  const { data: documents } = useDocuments();

  const totalChunks =
    documents?.reduce((acc, doc) => acc + doc.chunks, 0) ?? 0;
  const totalSize =
    documents?.reduce((acc, doc) => acc + (doc.file_size_bytes ?? 0), 0) ?? 0;

  const modelLabel =
    MODEL_OPTIONS.find((m) => m.value === settings.model)?.label ??
    settings.model;

  return (
    <aside
      className={cn(
        "flex h-screen flex-col bg-[#24292d] text-[#F3F4F6] transition-all duration-200",
        expanded ? "w-60" : "w-16"
      )}
    >
      {/* Hamburger toggle — always visible at top */}
      <div className={cn(
        "flex items-center px-3 py-3 border-b border-[#3a4149]",
        expanded ? "justify-between" : "justify-center"
      )}>
        {expanded && (
          <span className="text-sm font-bold tracking-tight text-white">
            RAG <span className="text-[#2fbb4f]">Q&A</span>
          </span>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          onClick={() => setExpanded((prev) => !prev)}
        >
          {expanded ? (
            <X className="size-4" />
          ) : (
            <Menu className="size-4" />
          )}
        </Button>
      </div>

      {/* Nav items */}
      <nav className="flex flex-col gap-1 p-2 pt-3">
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
                    : "text-[#8b949e] hover:text-white hover:bg-[#2b3137] border-l-2 border-transparent"
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
      <div className="mx-3 my-2 h-px bg-[#3a4149]" />

      {/* Settings section — only when expanded */}
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
                    className="w-full justify-start"
                  />
                }
              >
                {modelLabel}
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuRadioGroup
                  value={settings.model}
                  onValueChange={(value) => update({ model: value as string })}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <DropdownMenuRadioItem key={opt.value} value={opt.value}>
                      {opt.label}
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
                update({ topK: v });
              }}
            />
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Divider line */}
      <div className="mx-3 my-1 h-px bg-[#3a4149]" />

      {/* Collection stats — only when expanded */}
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
    </aside>
  );
}
