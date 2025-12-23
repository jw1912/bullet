use super::*;

#[test]
fn make_elementwise() {
    let mut elmt = ElementwiseDescription::default();
    let a = elmt.add_input(DType::F32);
    let b = elmt.add_input(DType::F32);

    let c = elmt.binary(a, b, Binary::Add).unwrap();

    assert_eq!(elmt.len(), 3);
    assert_eq!(elmt.roots(), 1);
    assert_eq!(elmt.leaves(), 2);
    assert!(elmt.is_root(c));
}

#[test]
fn merge() {
    let mut elmt1 = ElementwiseDescription::default();
    let a = elmt1.add_input(DType::F32);
    let b = elmt1.add_input(DType::F32);

    let c = elmt1.binary(a, b, Binary::Add).unwrap();

    let mut elmt2 = ElementwiseDescription::default();
    let c2 = elmt2.add_input(DType::F32);
    let d = elmt2.add_input(DType::F32);

    let e = elmt2.binary(c2, d, Binary::Add).unwrap();

    let elmt = elmt1.merge_with(&elmt2, &[(c, c2)]).unwrap();

    assert_eq!(elmt.len(), 5);
    assert_eq!(elmt.roots(), 1);
    assert_eq!(elmt.leaves(), 3);
    assert!(elmt.is_root(e));

    let mut expected = ElementwiseDescription::default();
    let exp_a = expected.add_input(DType::F32);
    let exp_b = expected.add_input(DType::F32);
    let exp_c = expected.binary(exp_a, exp_b, Binary::Add).unwrap();
    let exp_d = expected.add_input(DType::F32);
    let exp_e = expected.binary(exp_c, exp_d, Binary::Add).unwrap();

    expected.relabel(&[(exp_a, a), (exp_b, b), (exp_c, c), (exp_d, d), (exp_e, e)]).unwrap();

    assert_eq!(elmt, expected);
}

#[test]
fn invalid_merge() {
    let mut elmt1 = ElementwiseDescription::default();
    let a = elmt1.add_input(DType::F32);
    let b = elmt1.add_input(DType::F32);

    let c = elmt1.binary(a, b, Binary::Add).unwrap();

    let mut elmt2 = ElementwiseDescription::default();
    let c2 = elmt2.add_input(DType::F32);
    let d = elmt2.add_input(DType::F32);

    let e = elmt2.binary(c2, d, Binary::Add).unwrap();

    assert!(elmt1.merge_with(&elmt2, &[(c, e)]).is_none());
}

#[test]
fn evaluate() {
    let mut elmt = ElementwiseDescription::default();
    let fp_a = elmt.add_input(DType::F32);
    let fp_b = elmt.add_input(DType::F32);

    let fp_c = elmt.binary(fp_a, fp_b, Binary::Add).unwrap();

    let int_a = elmt.add_input(DType::I32);
    let int_b = elmt.add_input(DType::I32);

    let int_c = elmt.binary(int_a, int_b, Binary::Add).unwrap();

    let fp_int_c = elmt.unary(int_c, Unary::Cast(DType::F32)).unwrap();

    let out = elmt.binary(fp_c, fp_int_c, Binary::Div).unwrap();

    let inputs = [
        (fp_a, DTypeValue::F32(1.0)),
        (fp_b, DTypeValue::F32(2.0)),
        (int_a, DTypeValue::I32(1)),
        (int_b, DTypeValue::I32(1)),
    ]
    .into();
    let values = elmt.evaluate(inputs, [out]).unwrap();

    assert_eq!(values.len(), 1);
    assert_eq!(values[0], DTypeValue::F32(1.5));
}

#[test]
fn evaluate_invalid_input() {
    let mut elmt = ElementwiseDescription::default();
    let a = elmt.add_input(DType::F32);
    let b = elmt.add_input(DType::F32);

    let c = elmt.binary(a, b, Binary::Add).unwrap();

    let inputs = [(a, DTypeValue::F32(1.0)), (b, DTypeValue::I32(1))].into();
    assert!(elmt.evaluate(inputs, [c]).is_none());
}
