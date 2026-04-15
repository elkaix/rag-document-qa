import type { DocumentInfo } from "@/api/types";
import { Card, CardContent } from "@/components/ui/card";

interface DocStatsProps {
  documents: DocumentInfo[];
}

function formatMB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(2);
}

export function DocStats({ documents }: DocStatsProps) {
  const totalChunks = documents.reduce((acc, doc) => acc + doc.chunks, 0);
  const totalSize = documents.reduce(
    (acc, doc) => acc + (doc.file_size_bytes ?? 0),
    0
  );
  const fileTypes = new Set(
    documents.map((d) => d.file_type).filter(Boolean)
  ).size;

  const stats = [
    { label: "Documents", value: documents.length },
    { label: "Chunks", value: totalChunks },
    { label: "Total Size (MB)", value: formatMB(totalSize) },
    { label: "File Types", value: fileTypes },
  ];

  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {stats.map((stat) => (
        <Card key={stat.label} size="sm">
          <CardContent className="flex flex-col items-center gap-1 py-2">
            <span className="text-3xl font-bold tabular-nums">
              {stat.value}
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {stat.label}
            </span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
