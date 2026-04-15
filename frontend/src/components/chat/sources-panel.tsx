import { useState } from "react";
import { FileText } from "lucide-react";
import type { SourceInfo } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SourcesPanelProps {
  sources: SourceInfo[];
}

// TRADE-OFF: ChromaDB cosine similarity scores are typically 0.2–0.6 for
//            dense embeddings (all-MiniLM-L6-v2). Scores above 0.5 are
//            strong matches. The old TF-IDF thresholds (0.85/0.5) were too
//            high — everything showed as red/destructive.
function scoreColor(score: number): "default" | "secondary" | "destructive" {
  if (score >= 0.45) return "default";
  if (score >= 0.25) return "secondary";
  return "destructive";
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(
    sources.length > 0 ? 0 : null
  );

  if (sources.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center text-muted-foreground px-4">
        <FileText className="size-12 opacity-30" />
        <p className="text-sm">Sources will appear here after a query.</p>
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1">
      <div className="flex flex-col gap-2 p-4">
        {sources.map((source, idx) => (
          <Collapsible
            key={`${source.doc_id}-${source.chunk_id}`}
            open={expandedIndex === idx}
            onOpenChange={(open) => setExpandedIndex(open ? idx : null)}
          >
            <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm rounded-lg border border-[#E5E7EB] bg-white hover:border-[#0d74e7] hover:bg-[#EBF3FE] transition-all">
              <FileText className="size-4 shrink-0 text-muted-foreground" />
              <span className="flex-1 truncate">
                {source.filename ?? "Unknown file"}
              </span>
              <Badge variant={scoreColor(source.score)}>
                {source.score.toFixed(2)}
              </Badge>
            </CollapsibleTrigger>
            <CollapsibleContent className="px-3 py-2 mt-1">
              <p className="whitespace-pre-wrap p-3 text-xs leading-relaxed text-muted-foreground rounded-lg bg-[#F9FAFB] border-l-2 border-[#0d74e7]">
                {source.excerpt}
              </p>
              <div className="mt-2 flex gap-3 text-[10px] text-muted-foreground/70">
                <span>doc: {source.doc_id.slice(0, 8)}</span>
                <span>chunk: {source.chunk_id.slice(0, 8)}</span>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </div>
    </ScrollArea>
  );
}
