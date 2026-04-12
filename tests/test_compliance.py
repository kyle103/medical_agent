from app.core.compliance.compliance_service import ComplianceService


def test_input_sensitive_info_block():
    ok, msg = ComplianceService().input_compliance_check("身份证号 11010119900307876X")
    assert ok is False
    assert "敏感" in msg


def test_output_forbidden_word_block():
    ok, msg = ComplianceService().output_compliance_check("建议你确诊后再治疗")
    assert ok is False
