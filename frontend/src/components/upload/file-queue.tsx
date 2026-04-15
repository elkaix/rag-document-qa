import type { FileQueueItem } from "@/hooks/use-upload";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface FileQueueProps {
  items: FileQueueItem[];
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileExtension(name: string): string {
  const parts = name.split(".");
  return parts.length > 1 ? parts[parts.length - 1].toUpperCase() : "FILE";
}

function StatusCell({ item }: { item: FileQueueItem }) {
  switch (item.status) {
    case "pending":
      return <Badge variant="outline">Pending</Badge>;
    case "uploading":
      return (
        <div className="w-24">
          <Progress value={null} />
        </div>
      );
    case "done":
      return <Badge variant="secondary">Indexed</Badge>;
    case "error":
      return (
        <Badge variant="destructive" title={item.error}>
          Error
        </Badge>
      );
  }
}

export function FileQueue({ items }: FileQueueProps) {
  if (items.length === 0) return null;

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Filename</TableHead>
          <TableHead>Size</TableHead>
          <TableHead>Type</TableHead>
          <TableHead>Status</TableHead>
          <TableHead className="text-right">Chunks</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {items.map((item, idx) => (
          <TableRow key={`${item.file.name}-${idx}`}>
            <TableCell className="max-w-[200px] truncate font-medium">
              {item.file.name}
            </TableCell>
            <TableCell>{formatSize(item.file.size)}</TableCell>
            <TableCell>
              <Badge variant="outline">{fileExtension(item.file.name)}</Badge>
            </TableCell>
            <TableCell>
              <StatusCell item={item} />
            </TableCell>
            <TableCell className="text-right">
              {item.result?.chunks_count ?? "—"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
