"""CFD surrogate stage that predicts mouth-throat deposition metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Set

from ...contracts.stage import Stage
from ...contracts.types import CFDInput, CFDResult
from ...contracts.errors import ModelError
from ...models.cfd.ml_models import ML_CFD_MT_deposition


@dataclass
class MLCFDStage(Stage[CFDInput, CFDResult]):
    """Stage wrapper around the machine-learned CFD surrogate."""

    name: str = "ml_cfd"
    provides: Set[str] = frozenset({"cfd"})
    requires: Set[str] = frozenset()

    def run(self, data: CFDInput) -> CFDResult:
        try:
            subject = data.subject
            product = data.product
            maneuver = data.maneuver
            api = data.api

            if subject is None or product is None or maneuver is None:
                raise ModelError("CFD stage requires subject, product, and maneuver entities")
            if api is None:
                raise ModelError("CFD stage requires API entity")

            # Resolve final product parameters for this API
            product_final = product.get_final_values(getattr(api, "name", None))
            mmad = product_final._final_mmad
            gsd = product_final._final_gsd
            propellant = getattr(product_final, "propellant", None)
            usp_depo_fraction = getattr(product_final, "_final_usp_depo_fraction", None)
            pifr = getattr(maneuver, "pifr_Lpm", None)

            if pifr is None or pifr <= 0:
                raise ModelError("CFD stage requires a positive peak inspiratory flow rate")
            if usp_depo_fraction is None:
                raise ModelError("CFD stage requires USP deposition fraction on the product")

            params: Mapping[str, Any] = data.params or {}
            cast = str(params.get("cast", "medium") or "medium").strip().lower()
            device_type = str(
                params.get("device_type", getattr(product_final, "device", "dfp"))
                or "dfp"
            ).strip().lower()
            propellant_key = str(propellant or "").strip().upper() or None
            data_root = params.get("data_root")
            database_filename = params.get("database_filename", "DataBase_July_10.csv")
            model_root = params.get("model_root")

            mt_fraction, mmad_cast, gsd_cast = ML_CFD_MT_deposition(
                mmad=mmad,
                gsd=gsd,
                propellant=propellant_key,
                usp_deposition=usp_depo_fraction,
                pifr=pifr,
                cast=cast,
                DeviceType=device_type,
                data_root=data_root,
                database_filename=database_filename,
                model_root=model_root,
            )

            metadata = {
                "propellant": propellant_key,
                "usp_deposition": usp_depo_fraction,
                "pifr_Lpm": pifr,
                "cast": cast,
                "device_type": device_type,
                "database": database_filename,
            }

            return CFDResult(
                mmad=mmad_cast,
                gsd=gsd_cast,
                mt_deposition_fraction=mt_fraction,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelError(f"CFD stage failed: {exc}") from exc
