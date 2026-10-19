import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from hostrada4py import extract_diffuse_radiation_for_point

    return (extract_diffuse_radiation_for_point,)


@app.cell
def _():
    YEAR = 2024
    return (YEAR,)


@app.cell
def _():
    LONGITUDE = 13.405
    return (LONGITUDE,)


@app.cell
def _():
    LATITUDE = 52.52
    return (LATITUDE,)


@app.cell
def _(LATITUDE, LONGITUDE, YEAR, extract_diffuse_radiation_for_point):
    df = extract_diffuse_radiation_for_point(
        lon=LONGITUDE,
        lat=LATITUDE,
        start=f"{YEAR}-01-01T00:00:00",
        end=f"{YEAR}-12-31T23:00:00",
        tz="Europe/Berlin",
    )
    df.index = df.time
    return (df,)


@app.cell
def _(df):
    df.dni.describe()
    return


@app.cell
def _():
    import seaborn as sns

    return (sns,)


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(LATITUDE, LONGITUDE, YEAR, df, pd, plt, sns):
    ax=sns.heatmap(
        pd.pivot_table(
            df, values="dni", index=df.index.time, columns=df.index.dayofyear
        ),
        annot=False,
        cmap="viridis",
        cbar_kws={'format': '%g W / m²'}
        #vmin=500,
        #vmax=1000,
    )
    ax.invert_yaxis()
    plt.title(f"DNI at {LATITUDE}°N {LONGITUDE}°E in {YEAR}\nNote: Corrected, and well centered on midday")
    plt.savefig('correct_dni.png')
    ax
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
