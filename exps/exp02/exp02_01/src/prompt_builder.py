import pandas as pd
import numpy as np


def _round_to_10(value: float) -> int:
    """残業時間を10の倍数に丸める"""
    return int(round(value / 10) * 10)


def build_prompt(
    employee_id: str,
    category: str,
    df_overtime: pd.DataFrame,
    df_position: pd.DataFrame,
    df_dx: pd.DataFrame,
    df_hr: pd.DataFrame,
    df_udemy: pd.DataFrame,
) -> str:
    """1名分のテーブルデータをテキストプロンプトに変換する。

    Args:
        employee_id: 社員番号
        category: 社員のcategory（train/testのcategoryカラム）
        df_overtime: 前処理済み overtime_work_by_month
        df_position: 前処理済み position_history
        df_dx: 前処理済み dx
        df_hr: 前処理済み hr（実施開始日カラムが必要）
        df_udemy: 前処理済み udemy_activity

    Returns:
        テキストプロンプト文字列
    """
    # ## 社員ごとのデータ抽出
    emp_ot = df_overtime[df_overtime['社員番号'] == employee_id].copy()
    emp_pos = df_position[df_position['社員番号'] == employee_id].copy()
    emp_dx = df_dx[df_dx['社員番号'] == employee_id].copy()
    emp_hr = df_hr[df_hr['社員番号'] == employee_id].copy()
    emp_udemy = df_udemy[df_udemy['社員番号'] == employee_id].copy()

    # ## 年カラム追加
    if not emp_ot.empty:
        emp_ot = emp_ot.copy()
        emp_ot['_year'] = emp_ot['date'].dt.year

    if not emp_pos.empty:
        emp_pos = emp_pos.copy()
        # yearは2桁（例: 22→2022）
        emp_pos['_year'] = emp_pos['year'].apply(lambda y: 2000 + int(y))

    if not emp_dx.empty:
        emp_dx = emp_dx.copy()
        emp_dx['_year'] = emp_dx['研修実施日'].dt.year

    if not emp_hr.empty:
        emp_hr = emp_hr.copy()
        emp_hr['_year'] = emp_hr['実施開始日'].dt.year

    if not emp_udemy.empty:
        emp_udemy = emp_udemy.copy()
        emp_udemy['_year'] = emp_udemy['開始日'].dt.year

    # ## 対象年の収集
    list_years = set()
    for df in [emp_ot, emp_pos, emp_dx, emp_hr, emp_udemy]:
        if not df.empty and '_year' in df.columns:
            list_years.update(df['_year'].dropna().astype(int).unique())

    # ## プロンプト構築
    lines = [category, ""]

    for year in sorted(list_years):
        lines.append(f"{year}年")

        # 役職情報
        year_pos = emp_pos[emp_pos['_year'] == year] if not emp_pos.empty else pd.DataFrame()
        for _, row in year_pos.iterrows():
            lines.append(f"{row['勤務区分']}({row['役職']})")

        # 残業情報（月次集計→年次平均/最大）
        year_ot = emp_ot[emp_ot['_year'] == year] if not emp_ot.empty else pd.DataFrame()
        if not year_ot.empty:
            avg_ot = year_ot['hours'].mean()
            max_ot = year_ot['hours'].max()
            lines.append(f"平均残業時間: 約{_round_to_10(avg_ot)}時間")
            lines.append(f"最大残業時間: 約{_round_to_10(max_ot)}時間")

        # DX研修
        year_dx = emp_dx[emp_dx['_year'] == year] if not emp_dx.empty else pd.DataFrame()
        for _, row in year_dx.iterrows():
            lines.append(f"{row['研修カテゴリ']}({row['研修名']})")

        # HR施策
        year_hr = emp_hr[emp_hr['_year'] == year] if not emp_hr.empty else pd.DataFrame()
        for _, row in year_hr.iterrows():
            lines.append(f"{row['カテゴリ']}({row['研修名']})")

        # Udemyコース（コースIDで重複排除）
        year_udemy = emp_udemy[emp_udemy['_year'] == year] if not emp_udemy.empty else pd.DataFrame()
        for _, row in year_udemy.drop_duplicates(subset=['コースID']).iterrows():
            lines.append(f"{row['コースカテゴリー']}({row['コースタイトル']})")

        lines.append("")

    return "\n".join(lines).strip()


def build_prompt_df(
    df_base: pd.DataFrame,
    df_overtime: pd.DataFrame,
    df_position: pd.DataFrame,
    df_dx: pd.DataFrame,
    df_hr: pd.DataFrame,
    df_udemy: pd.DataFrame,
) -> pd.DataFrame:
    """train/testデータフレーム全体のプロンプトを一括生成する。

    Args:
        df_base: 社員番号・categoryカラムを持つDataFrame（train or test）
        df_overtime: 前処理済み overtime_work_by_month
        df_position: 前処理済み position_history
        df_dx: 前処理済み dx
        df_hr: 前処理済み hr
        df_udemy: 前処理済み udemy_activity

    Returns:
        社員番号・category・promptカラムを持つDataFrame
    """
    list_records = []
    for _, row in df_base.iterrows():
        prompt = build_prompt(
            employee_id=row['社員番号'],
            category=row['category'],
            df_overtime=df_overtime,
            df_position=df_position,
            df_dx=df_dx,
            df_hr=df_hr,
            df_udemy=df_udemy,
        )
        list_records.append({'社員番号': row['社員番号'], 'category': row['category'], 'prompt': prompt})

    return pd.DataFrame(list_records)
