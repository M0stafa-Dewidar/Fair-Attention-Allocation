import pandas as pd
import matplotlib.pyplot as plt





def main():
    df = pd.read_csv("Crime_data_from_2020_to_Present_20251116.csv")
    print("got here!")
    print(df.columns)
    slice = df[["AREA NAME", "AREA", "DATE OCC"]]
    # print(slice)
    incident_rate_by_areas = slice.groupby("AREA NAME").size()
    print(type(incident_rate_by_areas))
    incident_rate_by_areas /= 2145
    incident_rate_by_areas.sort_values(inplace=True)
    incident_rate_by_areas.plot.bar()
    plt.show()
    # print(incident_rate_by_areas)
    # plt.bar(incident_rate_by_areas[0], incident_rate_by_areas[1])




if __name__ == "__main__":
    main()