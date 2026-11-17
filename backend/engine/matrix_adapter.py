"""
Industrial Matrix Workbook & Multi-Sheet History Card Adapter.

Automatically detects, harmonizes, and unpivots multi-sheet industrial logbooks,
reduction cell potline history cards, and equipment parameter workbooks where
data is cross-tabulated across multiple worksheets (Units x Days/Months).
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
import openpyxl
import pandas as pd
import numpy as np


class IndustrialMatrixAdapter:
    """
    Universal adapter for industrial multi-sheet parameter matrix workbooks.
    Converts 2D cross-tabulated worksheets into normalized longitudinal datasets.
    """

    # Parameter normalization mapping
    PARAMETER_MAP = {
        'working volt': ('Working_Voltage_V', 'Cell Operating Voltage (Volts)'),
        'set volt': ('Set_Voltage_V', 'Target Setpoint Voltage (Volts)'),
        'avg.volt': ('Average_Voltage_V', 'Average Cell Voltage (Volts)'),
        'gross.volt': ('Gross_Voltage_V', 'Gross Line Voltage (Volts)'),
        'b.temp': ('Bath_Temperature_C', 'Electrolyte Bath Temperature (°C)'),
        'mh': ('Metal_Height_cm', 'Molten Metal Pad Height (cm)'),
        'bh': ('Bath_Height_cm', 'Cryolite Bath Height (cm)'),
        'base mh': ('Base_Metal_Height_cm', 'Baseline Metal Height (cm)'),
        'act ce%': ('Current_Efficiency_Pct', 'Faradaic Current Efficiency (%)'),
        'eff': ('Nominal_Efficiency_Pct', 'Nominal Current Efficiency (%)'),
        'eff (2)': ('Secondary_Efficiency_Pct', 'Secondary Efficiency (%)'),
        'dc energy': ('DC_Energy_kWh_t', 'Specific DC Energy Consumption (kWh/t Al)'),
        'noise': ('MHD_Noise_mV', 'Interfacial Magnetohydrodynamic Noise (mV)'),
        'shaky dur': ('Shaky_Duration_min', 'Cell Instability / Shaky Duration (min)'),
        'nb': ('Narrow_Band_Noise_mV', 'Narrow Band Noise (mV)'),
        'alf3 dossage': ('AlF3_Dosage_kg', 'AlF3 Chemical Addition Dosage (kg)'),
        'ex.alf3': ('Excess_AlF3_Pct', 'Excess Aluminium Fluoride (%)'),
        'soda': ('Soda_Dosage_kg', 'Soda Ash Addition (kg)'),
        'alumina': ('Alumina_Dumps', 'Point Feeder Alumina Dumps (count)'),
        'tapped': ('Tapped_Cycles', 'Metal Siphon Tapping Cycles (count)'),
        'tapped (wt)': ('Tapped_Weight_kg', 'Tapped Metal Weight (kg)'),
        'fe': ('Iron_Fe_Pct', 'Iron Impurity (Fe %)'),
        'si': ('Silicon_Si_Pct', 'Silicon Impurity (Si %)'),
        'mn': ('Manganese_Mn_Pct', 'Manganese Impurity (Mn %)'),
        'cr': ('Chromium_Cr_Pct', 'Chromium Impurity (Cr %)'),
        'cu': ('Copper_Cu_Pct', 'Copper Impurity (Cu %)'),
        'actual-e': ('Anode_Effects_Count', 'Anode Effect Excursions (count)'),
        'ae': ('Anode_Effects_Nominal', 'Anode Effects (count)'),
        'aei': ('Anode_Effect_Intensity', 'Anode Effect Intensity (V*min)'),
        'aed': ('Anode_Effect_Duration', 'Anode Effect Duration (min)'),
        'ka': ('Line_Current_kA', 'Potline Series Amperage (kA)')
    }

    @classmethod
    def is_industrial_matrix_workbook(cls, file_path: str) -> bool:
        """
        Determines if an Excel workbook matches the industrial multi-parameter matrix pattern.
        """
        if not file_path or not os.path.exists(file_path):
            return False
        if not file_path.lower().endswith(('.xlsx', '.xls')):
            return False

        try:
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            names = [s.lower().strip() for s in wb.sheetnames]
            wb.close()

            # Rule 1: High sheet count with typical industrial keywords
            matched_params = sum(1 for p in cls.PARAMETER_MAP.keys() if p in names)
            if matched_params >= 3:
                return True

            # Rule 2: Explicit industrial indicators
            has_report = any(k in names for k in ['report', 'pot history', 'history card'])
            has_age = any(k in names for k in ['age', 'actual age', 'lining drop'])
            if has_report and (has_age or matched_params >= 2):
                return True

            return False
        except Exception:
            return False

    @classmethod
    def parse_workbook(cls, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Parses an industrial matrix workbook into clean, relational DataFrames:
        1. 'Master_Potline_TimeSeries': Normalized cell-day observations.
        2. 'Pot_Fleet_Summary': Asset-level aggregation (1 row per pot).
        3. 'Cathode_Lining_History': Monthly degradation time-series.
        """
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet_names_lower = {s.lower().strip(): s for s in wb.sheetnames}

        # 1. Parse Asset Metadata (from 'AGE' or similar sheet)
        pot_meta = cls._parse_asset_metadata(wb, sheet_names_lower)

        # 2. Parse Daily Parameter Sheets
        parsed_daily, reporting_month = cls._parse_daily_matrices(wb, sheet_names_lower)

        # 3. Parse Monthly Degradation ('Lining drop')
        monthly_ld, df_lining_history = cls._parse_lining_drop(wb, sheet_names_lower)

        # 4. Assemble Master Time-Series DataFrame
        df_master = cls._assemble_master_timeseries(
            parsed_daily, pot_meta, monthly_ld, reporting_month
        )

        # 5. Assemble Pot Fleet Summary Table
        df_summary = cls._assemble_pot_summary(df_master, pot_meta, monthly_ld)

        wb.close()

        result = {}
        if not df_master.empty:
            result['Potline_Master_TimeSeries'] = df_master
        if not df_summary.empty:
            result['Pot_Fleet_Summary'] = df_summary
        if df_lining_history is not None and not df_lining_history.empty:
            result['Cathode_Lining_History'] = df_lining_history

        return result

    @classmethod
    def _parse_asset_metadata(cls, wb, sheet_names_lower) -> Dict[int, Dict[str, Any]]:
        pot_meta = {}
        age_sheet_name = None
        for k in ['age', 'actual age', 'pot age']:
            if k in sheet_names_lower:
                age_sheet_name = sheet_names_lower[k]
                break

        if not age_sheet_name:
            return pot_meta

        ws = wb[age_sheet_name]
        rows = list(ws.iter_rows(max_row=350, max_col=35, values_only=True))
        if not rows:
            return pot_meta

        header_row = 3
        pot_col = 1
        for r_idx, row in enumerate(rows[:8]):
            for c_idx, val in enumerate(row[:6]):
                if val and 'POT NO' in str(val).upper():
                    header_row = r_idx
                    pot_col = c_idx
                    break

        for row in rows[header_row + 1:]:
            if len(row) <= pot_col or row[pot_col] is None:
                continue
            pot_val = row[pot_col]
            if not str(pot_val).strip().isdigit():
                continue
            pot_no = int(pot_val)
            
            start_d = row[pot_col + 1] if len(row) > pot_col + 1 else None
            restart_d = row[pot_col + 2] if len(row) > pot_col + 2 else None
            age_d = row[pot_col + 3] if len(row) > pot_col + 3 else None
            restarts = row[pot_col + 5] if len(row) > pot_col + 5 else None
            
            l_type = row[27] if len(row) > 27 else None
            vendor = row[28] if len(row) > 28 else None

            try:
                age_clean = float(age_d) if age_d is not None else np.nan
            except (ValueError, TypeError):
                age_clean = np.nan

            try:
                restarts_clean = int(restarts) if restarts is not None else 0
            except (ValueError, TypeError):
                restarts_clean = 0

            pot_meta[pot_no] = {
                'Pot_No': pot_no,
                'Start_Date': str(start_d)[:10] if start_d else None,
                'Pot_Age_Days': age_clean,
                'Restart_Count': restarts_clean,
                'Lining_Type': str(l_type).strip() if l_type else 'Standard Cathode',
                'Cathode_Vendor': str(vendor).strip() if vendor else 'Standard Supplier'
            }

        return pot_meta

    @classmethod
    def _parse_daily_matrices(cls, wb, sheet_names_lower) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], str]:
        parsed_daily = {}
        reporting_month = "2026-09"

        for p_key, (feat_col, _) in cls.PARAMETER_MAP.items():
            if p_key not in sheet_names_lower:
                continue
            actual_name = sheet_names_lower[p_key]
            ws = wb[actual_name]
            rows = list(ws.iter_rows(max_row=350, max_col=40, values_only=True))
            if not rows:
                continue

            # Extract reporting month if available in row index 1 (row 2)
            if len(rows) > 1:
                for c_val in rows[1][:5]:
                    if c_val:
                        m_str = str(c_val)[:7]
                        if re.match(r'\d{4}-\d{2}', m_str):
                            reporting_month = m_str
                            break

            # Dynamically identify the row containing day column headers
            header_row = None
            day_cols = {}
            pot_col = 1

            for r_idx, row in enumerate(rows[:8]):
                found_cols = {}
                for c_idx, v in enumerate(row[:40]):
                    if v is not None:
                        v_str = str(v).strip()
                        m = re.findall(r'^\d+s?$', v_str)
                        if m:
                            num = int(re.findall(r'\d+', m[0])[0])
                            if 1 <= num <= 31:
                                found_cols[c_idx] = num
                if len(found_cols) >= 5:
                    header_row = r_idx
                    day_cols = found_cols
                    # Look for pot column label preceding the day columns
                    for c_idx in range(min(found_cols.keys())):
                        v_head = str(row[c_idx] or '').upper()
                        v_prev = str(rows[r_idx - 1][c_idx] or '').upper() if r_idx > 0 else ''
                        if 'POT' in v_head or 'POT' in v_prev:
                            pot_col = c_idx
                            break
                    break

            if header_row is None or not day_cols:
                continue

            # Read pot rows
            for row in rows[header_row + 1:]:
                if len(row) <= pot_col or row[pot_col] is None:
                    continue
                p_val = row[pot_col]
                if not str(p_val).strip().isdigit():
                    continue
                pot_no = int(p_val)

                for c_idx, d_num in day_cols.items():
                    if len(row) > c_idx:
                        val = row[c_idx]
                        if val is not None and val != '-' and val != '':
                            try:
                                val_flt = float(val)
                                key = (pot_no, d_num)
                                if key not in parsed_daily:
                                    parsed_daily[key] = {}
                                parsed_daily[key][feat_col] = val_flt
                            except (ValueError, TypeError):
                                pass

        return parsed_daily, reporting_month

    @classmethod
    def _parse_lining_drop(cls, wb, sheet_names_lower) -> Tuple[Dict[int, Dict[str, float]], Optional[pd.DataFrame]]:
        monthly_ld = {}
        lining_history_records = []

        lining_sheet_name = None
        for k in ['lining drop', 'lining_drop', 'cathode drop']:
            if k in sheet_names_lower:
                lining_sheet_name = sheet_names_lower[k]
                break

        if not lining_sheet_name:
            return monthly_ld, None

        ws = wb[lining_sheet_name]
        rows = list(ws.iter_rows(max_row=350, max_col=40, values_only=True))
        if not rows:
            return monthly_ld, None

        # Identify date columns in row index 3 (row 4)
        month_cols = {}
        if len(rows) > 3:
            for c_idx, cell_val in enumerate(rows[3]):
                if c_idx >= 2 and cell_val is not None:
                    c_str = str(cell_val)[:10]
                    if re.match(r'\d{4}-\d{2}', c_str):
                        month_cols[c_idx] = c_str

        for row in rows[4:]:
            if len(row) <= 1 or row[1] is None:
                continue
            pot_val = row[1]
            if not str(pot_val).strip().isdigit():
                continue
            pot_no = int(pot_val)
            age_val = row[0] if len(row) > 0 else None
            try:
                pot_age = float(age_val) if age_val is not None else np.nan
            except (ValueError, TypeError):
                pot_age = np.nan

            drops = []
            for c_idx, m_date in month_cols.items():
                if len(row) > c_idx:
                    v = row[c_idx]
                    if v is not None and v != '-' and v != '':
                        try:
                            v_flt = float(v)
                            drops.append(v_flt)
                            lining_history_records.append({
                                'Pot_No': pot_no,
                                'Evaluation_Month': m_date,
                                'Cathode_Lining_Drop_V': v_flt,
                                'Pot_Age_Days': pot_age
                            })
                        except (ValueError, TypeError):
                            pass

            if drops:
                monthly_ld[pot_no] = {
                    'Latest_Cathode_Lining_Drop_V': drops[-1],
                    'Avg_Cathode_Lining_Drop_V': float(np.mean(drops)),
                    'Max_Cathode_Lining_Drop_V': float(np.max(drops))
                }

        df_lining = pd.DataFrame(lining_history_records) if lining_history_records else None
        return monthly_ld, df_lining

    @classmethod
    def _assemble_master_timeseries(cls, parsed_daily, pot_meta, monthly_ld, reporting_month) -> pd.DataFrame:
        records = []
        for (pot_no, d_num), feats in parsed_daily.items():
            date_str = f"{reporting_month}-{d_num:02d}"
            row = {
                'Pot_No': pot_no,
                'Date': date_str,
                'Day_Of_Month': d_num
            }
            if pot_no in pot_meta:
                row.update(pot_meta[pot_no])
            if pot_no in monthly_ld:
                row.update(monthly_ld[pot_no])
            row.update(feats)
            records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Sort chronologically by Pot and Date
        if 'Pot_No' in df.columns and 'Date' in df.columns:
            df = df.sort_values(by=['Pot_No', 'Date']).reset_index(drop=True)

        return df

    @classmethod
    def _assemble_pot_summary(cls, df_master: pd.DataFrame, pot_meta: Dict[int, Dict[str, Any]], monthly_ld: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        if df_master.empty:
            return pd.DataFrame()

        num_cols = [c for c in df_master.select_dtypes(include=[np.number]).columns if c not in ['Pot_No', 'Day_Of_Month']]
        
        agg_rules = {}
        for c in num_cols:
            if c in ['Working_Voltage_V', 'Bath_Temperature_C', 'Metal_Height_cm', 'Bath_Height_cm']:
                agg_rules[c] = ['mean', 'min', 'max']
            elif c in ['Iron_Fe_Pct', 'Silicon_Si_Pct']:
                agg_rules[c] = ['mean', 'max']
            elif c in ['Current_Efficiency_Pct', 'DC_Energy_kWh_t', 'MHD_Noise_mV', 'AlF3_Dosage_kg']:
                agg_rules[c] = 'mean'
            elif c in ['Pot_Age_Days', 'Latest_Cathode_Lining_Drop_V', 'Avg_Cathode_Lining_Drop_V']:
                agg_rules[c] = 'first'

        if not agg_rules:
            return pd.DataFrame()

        df_summary = df_master.groupby('Pot_No').agg(agg_rules).reset_index()
        # Flatten MultiIndex columns
        df_summary.columns = [
            f"{c[0]}_{c[1]}" if isinstance(c, tuple) and c[1] and c[1] != 'first' else (c[0] if isinstance(c, tuple) else str(c))
            for c in df_summary.columns
        ]

        # Merge metadata
        if pot_meta:
            meta_rows = list(pot_meta.values())
            df_meta = pd.DataFrame(meta_rows)
            meta_cols = [c for c in df_meta.columns if c not in df_summary.columns or c == 'Pot_No']
            df_summary = df_summary.merge(df_meta[meta_cols], on='Pot_No', how='left')

        return df_summary
