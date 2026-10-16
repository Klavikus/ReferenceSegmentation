from pathlib import Path

import pandas as pd

from backend.contracts.pipeline_models import ExportArtifact, SegmentationExport

def write_segmentation_excels(
    rows: list[dict],
    output_dir: str | Path,
) -> SegmentationExport:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    object_table = pd.DataFrame(rows)

    object_table_path = output_path / "output.xlsx"
    object_table.to_excel(
        object_table_path,
        index=False,
        sheet_name="Сегментированные объекты",
    )
    if not object_table_path.is_file():
        raise ValueError("Error saving Excel file")

    overall_table = pd.DataFrame(
        {
            "Общее кол-во частиц": [len(rows)],
            "Средняя площадь (px^2)": [object_table["Площадь (px^2)"].mean()],
            "Средний периметр (px)": [object_table["Периметр (px)"].mean()],
            "Средняя площадь (nm^2)": [object_table["Площадь (nm^2)"].mean()],
            "Средний периметр (nm)": [object_table["Периметр (nm)"].mean()],
        }
    )
    overall_table_path = output_path / "overall.xlsx"
    overall_table.to_excel(
        overall_table_path,
        index=False,
        sheet_name="Общие данные",
    )
    if not overall_table_path.is_file():
        raise ValueError("Error saving overall Excel file")

    return SegmentationExport(
        object_table=object_table,
        object_table_path=str(object_table_path),
        overall_table=overall_table,
        overall_table_path=str(overall_table_path),
        artifacts=[
            ExportArtifact(
                artifact_type="object_table",
                path=object_table_path,
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            ExportArtifact(
                artifact_type="overall_table",
                path=overall_table_path,
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        ],
    )
