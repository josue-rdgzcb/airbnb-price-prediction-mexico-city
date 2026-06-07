import pandas as pd

# Map raw property_type strings into broader property groups
def map_property_type(pt: str) -> str:
    """
    Map a raw property_type string into a broader category.
    Categories: apartment, house, guesthouse, hotel, unique/nature, other.
    """
    pt = str(pt).lower()

    if any(x in pt for x in ["apartment", "condo", "loft", "rental unit", "aparthotel"]):
        return "apartment"
    elif any(x in pt for x in ["home", "townhouse", "villa", "cottage", "vacation home", "casa", "tiny home", "earthen home"]):
        return "house"
    elif any(x in pt for x in ["guesthouse", "guest suite"]):
        return "guesthouse"
    elif any(x in pt for x in ["hotel", "hostel", "bed and breakfast"]):
        return "hotel"
    elif any(x in pt for x in [
        "dome","hut","yurt","tent","barn","camper","farm stay","cabin","chalet",
        "bungalow","lighthouse","nature lodge","holiday park","resort","minsu",
        "shipping container","castle","tower"
    ]):
        return "unique/nature"
    else:
        return "other"

# Feature: group property types into broader categories
def add_property_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature 'property_group' by classifying
    property_type values into broader categories
    (e.g., apartment, house, hotel, unique/nature).

    """
    df = df.copy()
    
    df["property_group"] = df["property_type"].apply(map_property_type)
    return df


# Feature: combine property group with room type
def add_property_group_room(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite feature by combining property_group
    with the already normalized room_type.
    """

    df = df.copy()

    df["property_group_room"] = df["property_group"] + "_" + df["room_type"]
    return df


