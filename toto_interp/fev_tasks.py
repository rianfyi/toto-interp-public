from __future__ import annotations

from dataclasses import dataclass

# Registry derived from DataDog/toto `toto/evaluation/fev/tasks.yaml`
# and `toto/evaluation/fev/evaluate.py` (Apache-2.0).


@dataclass(frozen=True)
class FEVTaskSpec:
    dataset_config: str
    horizon: int
    num_windows: int
    seasonality: int
    target_fields: tuple[str, ...]
    known_dynamic_columns: tuple[str, ...] = ()
    past_dynamic_columns: tuple[str, ...] = ()
    static_columns: tuple[str, ...] = ()
    safe_for_paper: bool = False

    @property
    def exogenous_fields(self) -> tuple[str, ...]:
        return self.known_dynamic_columns + self.past_dynamic_columns


FEV_TASKS: dict[str, FEVTaskSpec] = {
    "proenfo_gfc12": FEVTaskSpec(
        dataset_config="proenfo_gfc12",
        horizon=168,
        num_windows=10,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("airtemperature",),
    ),
    "proenfo_gfc14": FEVTaskSpec(
        dataset_config="proenfo_gfc14",
        horizon=168,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("airtemperature",),
    ),
    "proenfo_gfc17": FEVTaskSpec(
        dataset_config="proenfo_gfc17",
        horizon=168,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("airtemperature",),
    ),
    "rohlik_sales_1D": FEVTaskSpec(
        dataset_config="rohlik_sales_1D",
        horizon=14,
        num_windows=1,
        seasonality=7,
        target_fields=("sales",),
        known_dynamic_columns=(
            "total_orders",
            "sell_price_main",
            "type_0_discount",
            "type_1_discount",
            "type_2_discount",
            "type_3_discount",
            "type_4_discount",
            "type_5_discount",
            "type_6_discount",
            "holiday",
            "shops_closed",
            "winter_school_holidays",
            "school_holidays",
        ),
        past_dynamic_columns=("availability",),
        static_columns=(
            "product_unique_id",
            "name",
            "L1_category_name_en",
            "L2_category_name_en",
            "L3_category_name_en",
            "L4_category_name_en",
            "warehouse",
        ),
        safe_for_paper=True,
    ),
    "rohlik_orders_1D": FEVTaskSpec(
        dataset_config="rohlik_orders_1D",
        horizon=61,
        num_windows=5,
        seasonality=7,
        target_fields=("orders",),
        known_dynamic_columns=("holiday", "shops_closed", "winter_school_holidays", "school_holidays"),
        past_dynamic_columns=(
            "shutdown",
            "mini_shutdown",
            "blackout",
            "mov_change",
            "frankfurt_shutdown",
            "precipitation",
            "snow",
            "user_activity_1",
            "user_activity_2",
        ),
        safe_for_paper=True,
    ),
    "entsoe_15T": FEVTaskSpec(
        dataset_config="entsoe_15T",
        horizon=96,
        num_windows=20,
        seasonality=96,
        target_fields=("target",),
        known_dynamic_columns=("radiation_diffuse_horizontal", "temperature", "radiation_direct_horizontal"),
        safe_for_paper=True,
    ),
    "entsoe_30T": FEVTaskSpec(
        dataset_config="entsoe_30T",
        horizon=96,
        num_windows=20,
        seasonality=48,
        target_fields=("target",),
        known_dynamic_columns=("radiation_diffuse_horizontal", "temperature", "radiation_direct_horizontal"),
        safe_for_paper=True,
    ),
    "entsoe_1H": FEVTaskSpec(
        dataset_config="entsoe_1H",
        horizon=168,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("radiation_diffuse_horizontal", "temperature", "radiation_direct_horizontal"),
        safe_for_paper=True,
    ),
    "epf_be": FEVTaskSpec(
        dataset_config="epf_be",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("Generation forecast", "System load forecast"),
        safe_for_paper=True,
    ),
    "epf_de": FEVTaskSpec(
        dataset_config="epf_de",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("PV+Wind Forecast", "Ampirion Load Forecast"),
        safe_for_paper=True,
    ),
    "epf_fr": FEVTaskSpec(
        dataset_config="epf_fr",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("Generation forecast", "System load forecast"),
        safe_for_paper=True,
    ),
    "epf_np": FEVTaskSpec(
        dataset_config="epf_np",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("Grid load forecast", "Wind power forecast"),
        safe_for_paper=True,
    ),
    "epf_pjm": FEVTaskSpec(
        dataset_config="epf_pjm",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("Zonal COMED load foecast", "System load forecast"),
        safe_for_paper=True,
    ),
    "rossmann_1D": FEVTaskSpec(
        dataset_config="rossmann_1D",
        horizon=48,
        num_windows=10,
        seasonality=7,
        target_fields=("Sales",),
        known_dynamic_columns=("SchoolHoliday", "Promo", "DayOfWeek", "Open", "StateHoliday"),
        past_dynamic_columns=("Customers",),
        static_columns=(
            "Store",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "PromoInterval",
        ),
    ),
    "m5_1D": FEVTaskSpec(
        dataset_config="m5_1D",
        horizon=28,
        num_windows=1,
        seasonality=7,
        target_fields=("target",),
        known_dynamic_columns=(
            "sell_price",
            "event_National",
            "event_Religious",
            "event_Cultural",
            "snap_CA",
            "event_Sporting",
            "snap_WI",
            "snap_TX",
        ),
        static_columns=("item_id", "dept_id", "cat_id", "store_id", "state_id"),
    ),
    "m5_1W": FEVTaskSpec(
        dataset_config="m5_1W",
        horizon=13,
        num_windows=1,
        seasonality=1,
        target_fields=("target",),
        known_dynamic_columns=(
            "sell_price",
            "event_National",
            "event_Religious",
            "event_Cultural",
            "snap_CA",
            "event_Sporting",
            "snap_WI",
            "snap_TX",
        ),
        static_columns=("item_id", "dept_id", "cat_id", "store_id", "state_id"),
    ),
    "favorita_stores_1D": FEVTaskSpec(
        dataset_config="favorita_stores_1D",
        horizon=28,
        num_windows=10,
        seasonality=7,
        target_fields=("sales",),
        known_dynamic_columns=("holiday", "onpromotion"),
        past_dynamic_columns=("oil_price",),
        static_columns=("store_nbr", "family", "city", "state", "type", "cluster"),
    ),
    "favorita_stores_1W": FEVTaskSpec(
        dataset_config="favorita_stores_1W",
        horizon=13,
        num_windows=10,
        seasonality=1,
        target_fields=("sales",),
        known_dynamic_columns=("onpromotion",),
        past_dynamic_columns=("oil_price",),
        static_columns=("store_nbr", "family", "city", "state", "type", "cluster"),
    ),
    "favorita_transactions_1D": FEVTaskSpec(
        dataset_config="favorita_transactions_1D",
        horizon=28,
        num_windows=10,
        seasonality=7,
        target_fields=("transactions",),
        known_dynamic_columns=("holiday",),
        past_dynamic_columns=("oil_price",),
        static_columns=("store_nbr", "city", "state", "type", "cluster"),
    ),
    "solar_with_weather_15T": FEVTaskSpec(
        dataset_config="solar_with_weather_15T",
        horizon=96,
        num_windows=20,
        seasonality=96,
        target_fields=("target",),
        known_dynamic_columns=("wind_speed", "day_length", "humidity", "rain_1h", "snow_1h", "temp", "pressure"),
        past_dynamic_columns=("global_horizontal_irradiance", "clouds_all"),
        safe_for_paper=True,
    ),
    "solar_with_weather_1H": FEVTaskSpec(
        dataset_config="solar_with_weather_1H",
        horizon=24,
        num_windows=20,
        seasonality=24,
        target_fields=("target",),
        known_dynamic_columns=("wind_speed", "day_length", "humidity", "rain_1h", "snow_1h", "temp", "pressure"),
        past_dynamic_columns=("global_horizontal_irradiance", "clouds_all"),
        safe_for_paper=True,
    ),
    "uci_air_quality_1H": FEVTaskSpec(
        dataset_config="uci_air_quality_1H",
        horizon=128,
        num_windows=20,
        seasonality=24,
        target_fields=("CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"),
        known_dynamic_columns=("T", "RH", "AH"),
        safe_for_paper=True,
    ),
    "uci_air_quality_1D": FEVTaskSpec(
        dataset_config="uci_air_quality_1D",
        horizon=28,
        num_windows=11,
        seasonality=7,
        target_fields=("CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"),
        known_dynamic_columns=("T", "RH", "AH"),
        safe_for_paper=True,
    ),
}


def get_fev_task(task_name: str) -> FEVTaskSpec | None:
    return FEV_TASKS.get(task_name)


def list_fev_tasks(*, safe_only: bool = False) -> list[FEVTaskSpec]:
    tasks = sorted(FEV_TASKS.values(), key=lambda task: task.dataset_config)
    if safe_only:
        tasks = [task for task in tasks if task.safe_for_paper]
    return tasks
