from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import Counter


@dataclass
class ProcessingStats:
    total: int
    found: int = 0
    partial: int = 0
    not_found: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.found + self.partial) / self.total * 100
    
    @property
    def found_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.found / self.total * 100
    
    @property
    def partial_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.partial / self.total * 100
    
    @property
    def not_found_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.not_found / self.total * 100


class StatisticsFormatter:
    
    def format(self, stats: ProcessingStats) -> str:
        lines = [
            "=" * 60,
            "Resultados de la búsqueda:",
            "=" * 60,
            f"   Total procesado: {stats.total}",
            f"   Encontrados: {stats.found} ({stats.found_rate:.1f}%)",
            f"   Parciales: {stats.partial} ({stats.partial_rate:.1f}%)",
            f"   No encontrados: {stats.not_found} ({stats.not_found_rate:.1f}%)",
            f"\n   Tasa de éxito: {stats.success_rate:.1f}%",
            "=" * 60
        ]
        return "\n".join(lines)


class DetailedStatisticsFormatter(StatisticsFormatter):
    
    def format(self, stats: ProcessingStats) -> str:
        lines = [
            "=" * 70,
            " RESULTADOS DE LA BÚSQUEDA:",
            "=" * 70,
            "",
            " Estadísticas Generales:",
            f"   Total procesado: {stats.total}",
            f"    Encontrados: {stats.found} ({stats.found_rate:.1f}%)",
            f"     Parciales: {stats.partial} ({stats.partial_rate:.1f}%)",
            f"    No encontrados: {stats.not_found} ({stats.not_found_rate:.1f}%)",
            "",
            f"    Tasa de éxito: {stats.success_rate:.1f}%"
        ]
        
        if stats.metadata.get('authors'):
            author_counts = Counter(stats.metadata['authors'])
            lines.append("")
            lines.append(" Autores Más Representados:")
            for author, count in author_counts.most_common(5):
                lines.append(f"   - {author}: {count} poemas")
        
        if stats.metadata.get('genres'):
            genre_counts = Counter(stats.metadata['genres'])
            lines.append("")
            lines.append(" Géneros Más Encontrados:")
            for genre, count in genre_counts.most_common(5):
                if genre != "N/A":
                    lines.append(f"   - {genre}: {count} poemas")
        
        if stats.metadata.get('content_types'):
            type_counts = Counter(stats.metadata['content_types'])
            lines.append("")
            lines.append(" Tipos de Contenido:")
            for content_type, count in type_counts.most_common():
                lines.append(f"   - {content_type}: {count} videos")
        
        if stats.metadata.get('qualities'):
            quality_counts = Counter(stats.metadata['qualities'])
            lines.append("")
            lines.append(" Distribución de Calidad:")
            quality_order = ['Excelente', 'Buena', 'Aceptable', 'Baja']
            for quality in quality_order:
                count = quality_counts.get(quality, 0)
                if count > 0:
                    lines.append(f"   - {quality}: {count} videos")
        
        if stats.metadata.get('duration_count', 0) > 0:
            avg_duration = stats.metadata['total_duration'] / stats.metadata['duration_count']
            lines.append("")
            lines.append(f"⏱  Duración Promedio: {avg_duration:.1f} minutos")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
