from ags_app.parser import parse_ags_text


def test_parse_ags_text_groups() -> None:
    content = b'''"GROUP","ISPT"\n"HEADING","LOCA_ID","ISPT_TOP","ISPT_MAIN"\n"UNIT",,"m",\n"TYPE","ID","2DP","0DP"\n"DATA","BH01","1.20","12"\n"DATA","BH01","2.00","18"\n'''

    tables = parse_ags_text(content)

    assert set(tables) == {"ISPT"}
    assert list(tables["ISPT"].columns) == ["LOCA_ID", "ISPT_TOP", "ISPT_MAIN"]
    assert len(tables["ISPT"]) == 2
    assert tables["ISPT"].iloc[1]["ISPT_MAIN"] == "18"


def test_parse_ags_text_detects_tab_delimiter() -> None:
    content = b"GROUP\tISPT\nHEADING\tLOCA_ID\tISPT_TOP\tISPT_MAIN\nDATA\tBH01\t1.20\t12\n"

    tables = parse_ags_text(content)

    assert tables["ISPT"].iloc[0].to_dict() == {
        "LOCA_ID": "BH01",
        "ISPT_TOP": "1.20",
        "ISPT_MAIN": "12",
    }
