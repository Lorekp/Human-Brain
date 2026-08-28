//! Minimal structural state for native typed-program holes.
//!
//! This module knows only the small typed surface language. It has no prompt,
//! task-family, token-ID, candidate-list, or temporal-runtime dependency. A
//! hole exposes both its transferable type/role cue and its constructor-neutral
//! root-to-hole child path, so locally identical holes can remain distinct in a
//! nested program without introducing task labels.

const MAX_LITERAL_DIGITS: u8 = 4;
const MAX_HOLE_ADDRESS_DEPTH: u8 = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedValue {
    Int,
    Bool,
    ListInt,
    IntToInt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceKind {
    Add,
    Absolute,
    LessThan,
    If,
    Map,
    Bind,
    Variable0,
    Integer,
    Plus,
    Minus,
    Digit(u8),
    End,
    Eos,
}

/// Physical cue identity for the next structural decision.
///
/// Expression cues contain only the expected value type. Literal cues contain
/// only their canonical phase. Neither records a parent operation, tree
/// position, prompt, or task family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedHoleCue {
    Expression(TypedValue),
    LiteralSign,
    LiteralFirstDigit,
    LiteralFirstNonzeroDigit,
    LiteralDigitOrEnd,
    LiteralEnd,
    Eos,
    Terminated,
}

/// Transferable structural position of the current expression/literal hole.
/// A binder body starts a new `Head`; constructor arguments use their generic
/// zero-based child ordinal. No primitive or task identity is represented.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedHoleRole {
    Head,
    Child0,
    Child1,
    Child2,
}

/// Compact constructor-neutral root-to-hole child path.
///
/// Each child ordinal occupies two bits; `depth` disambiguates leading and
/// trailing zero ordinals. The fixed capacity keeps the representation small
/// and makes unsupported depth fail closed rather than silently aliasing.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TypedHoleAddress {
    packed_ordinals: u16,
    depth: u8,
}

impl TypedHoleAddress {
    pub(crate) const fn root() -> Self {
        Self {
            packed_ordinals: 0,
            depth: 0,
        }
    }

    pub(crate) const fn depth(self) -> u8 {
        self.depth
    }

    pub(crate) const fn packed_ordinals(self) -> u16 {
        self.packed_ordinals
    }

    pub(crate) const fn child_ordinal(self, index: u8) -> Option<u8> {
        if index >= self.depth {
            return None;
        }
        Some(((self.packed_ordinals >> ((index as u32) * 2)) & 0b11) as u8)
    }

    fn child(self, ordinal: u8) -> Option<Self> {
        debug_assert!(
            ordinal <= 2,
            "typed constructors have at most three children"
        );
        if self.depth == MAX_HOLE_ADDRESS_DEPTH {
            return None;
        }
        Some(Self {
            packed_ordinals: self.packed_ordinals
                | (u16::from(ordinal) << (u32::from(self.depth) * 2)),
            depth: self.depth + 1,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedHoleError {
    UnsupportedUnarySignature {
        parameter: TypedValue,
        result: TypedValue,
    },
    Inadmissible {
        cue: TypedHoleCue,
        surface: SurfaceKind,
    },
    StructuralAddressOverflow {
        address: TypedHoleAddress,
        child_ordinal: u8,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Hole {
    expected: TypedValue,
    variable_zero: TypedValue,
    role: TypedHoleRole,
    address: TypedHoleAddress,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LiteralState {
    Sign,
    FirstDigit,
    FirstNonzeroDigit,
    Digits(u8),
    End,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TypedHoleState {
    holes: Vec<Hole>,
    literal: Option<LiteralState>,
    literal_role: TypedHoleRole,
    literal_address: TypedHoleAddress,
    terminated: bool,
}

impl TypedHoleState {
    pub(crate) fn unary(parameter: TypedValue, result: TypedValue) -> Result<Self, TypedHoleError> {
        if parameter == TypedValue::IntToInt || result == TypedValue::IntToInt {
            return Err(TypedHoleError::UnsupportedUnarySignature { parameter, result });
        }
        Ok(Self {
            holes: vec![Hole {
                expected: result,
                variable_zero: parameter,
                role: TypedHoleRole::Head,
                address: TypedHoleAddress::root(),
            }],
            literal: None,
            literal_role: TypedHoleRole::Head,
            literal_address: TypedHoleAddress::root(),
            terminated: false,
        })
    }

    pub(crate) fn cue(&self) -> TypedHoleCue {
        if self.terminated {
            return TypedHoleCue::Terminated;
        }
        if let Some(literal) = self.literal {
            return match literal {
                LiteralState::Sign => TypedHoleCue::LiteralSign,
                LiteralState::FirstDigit => TypedHoleCue::LiteralFirstDigit,
                LiteralState::FirstNonzeroDigit => TypedHoleCue::LiteralFirstNonzeroDigit,
                LiteralState::Digits(_) => TypedHoleCue::LiteralDigitOrEnd,
                LiteralState::End => TypedHoleCue::LiteralEnd,
            };
        }
        self.holes.last().map_or(TypedHoleCue::Eos, |hole| {
            TypedHoleCue::Expression(hole.expected)
        })
    }

    pub(crate) fn role(&self) -> TypedHoleRole {
        if self.literal.is_some() {
            return self.literal_role;
        }
        self.holes
            .last()
            .map_or(TypedHoleRole::Head, |hole| hole.role)
    }

    pub(crate) fn address(&self) -> TypedHoleAddress {
        if self.literal.is_some() {
            return self.literal_address;
        }
        self.holes
            .last()
            .map_or(TypedHoleAddress::root(), |hole| hole.address)
    }

    pub(crate) fn admits(&self, surface: SurfaceKind) -> bool {
        match (self.cue(), surface) {
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Add)
            | (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Absolute)
            | (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Integer)
            | (TypedHoleCue::Expression(TypedValue::Bool), SurfaceKind::LessThan)
            | (TypedHoleCue::Expression(_), SurfaceKind::If)
            | (TypedHoleCue::Expression(TypedValue::ListInt), SurfaceKind::Map)
            | (TypedHoleCue::Expression(TypedValue::IntToInt), SurfaceKind::Bind)
            | (TypedHoleCue::LiteralSign, SurfaceKind::Plus | SurfaceKind::Minus)
            | (TypedHoleCue::LiteralDigitOrEnd, SurfaceKind::End)
            | (TypedHoleCue::LiteralEnd, SurfaceKind::End)
            | (TypedHoleCue::Eos, SurfaceKind::Eos) => true,
            (TypedHoleCue::Expression(expected), SurfaceKind::Variable0) => self
                .holes
                .last()
                .is_some_and(|hole| hole.expected == expected && hole.variable_zero == expected),
            (TypedHoleCue::LiteralFirstDigit, SurfaceKind::Digit(digit)) => digit <= 9,
            (TypedHoleCue::LiteralFirstNonzeroDigit, SurfaceKind::Digit(digit)) => {
                (1..=9).contains(&digit)
            }
            (TypedHoleCue::LiteralDigitOrEnd, SurfaceKind::Digit(digit)) => digit <= 9,
            _ => false,
        }
    }

    pub(crate) fn advance(&mut self, surface: SurfaceKind) -> Result<(), TypedHoleError> {
        let cue = self.cue();
        if !self.admits(surface) {
            return Err(TypedHoleError::Inadmissible { cue, surface });
        }

        match (cue, surface) {
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Add) => {
                self.replace_current_hole([
                    (TypedValue::Int, TypedHoleRole::Child0),
                    (TypedValue::Int, TypedHoleRole::Child1),
                ])?;
            }
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Absolute) => {
                self.replace_current_hole([(TypedValue::Int, TypedHoleRole::Child0)])?;
            }
            (TypedHoleCue::Expression(TypedValue::Bool), SurfaceKind::LessThan) => {
                self.replace_current_hole([
                    (TypedValue::Int, TypedHoleRole::Child0),
                    (TypedValue::Int, TypedHoleRole::Child1),
                ])?;
            }
            (TypedHoleCue::Expression(result), SurfaceKind::If) => {
                self.replace_current_hole([
                    (TypedValue::Bool, TypedHoleRole::Child0),
                    (result, TypedHoleRole::Child1),
                    (result, TypedHoleRole::Child2),
                ])?;
            }
            (TypedHoleCue::Expression(TypedValue::ListInt), SurfaceKind::Map) => {
                self.replace_current_hole([
                    (TypedValue::IntToInt, TypedHoleRole::Child0),
                    (TypedValue::ListInt, TypedHoleRole::Child1),
                ])?;
            }
            (TypedHoleCue::Expression(TypedValue::IntToInt), SurfaceKind::Bind) => {
                self.replace_current_hole([(TypedValue::Int, TypedHoleRole::Head)])?;
                self.holes
                    .last_mut()
                    .expect("bind creates one body hole")
                    .variable_zero = TypedValue::Int;
            }
            (TypedHoleCue::Expression(_), SurfaceKind::Variable0) => {
                self.pop_hole();
            }
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Integer) => {
                let hole = self.pop_hole();
                self.literal_role = hole.role;
                self.literal_address = hole.address;
                self.literal = Some(LiteralState::Sign);
            }
            (TypedHoleCue::LiteralSign, SurfaceKind::Plus) => {
                self.literal = Some(LiteralState::FirstDigit);
            }
            (TypedHoleCue::LiteralSign, SurfaceKind::Minus) => {
                self.literal = Some(LiteralState::FirstNonzeroDigit);
            }
            (TypedHoleCue::LiteralFirstDigit, SurfaceKind::Digit(0)) => {
                self.literal = Some(LiteralState::End);
            }
            (
                TypedHoleCue::LiteralFirstDigit | TypedHoleCue::LiteralFirstNonzeroDigit,
                SurfaceKind::Digit(_),
            ) => {
                self.literal = Some(LiteralState::Digits(1));
            }
            (TypedHoleCue::LiteralDigitOrEnd, SurfaceKind::Digit(_)) => {
                let LiteralState::Digits(count) = self.literal.expect("digit phase exists") else {
                    unreachable!("digit-or-end cue owns a digit-count state")
                };
                self.literal = Some(if count + 1 == MAX_LITERAL_DIGITS {
                    LiteralState::End
                } else {
                    LiteralState::Digits(count + 1)
                });
            }
            (TypedHoleCue::LiteralDigitOrEnd | TypedHoleCue::LiteralEnd, SurfaceKind::End) => {
                self.literal = None;
            }
            (TypedHoleCue::Eos, SurfaceKind::Eos) => {
                self.terminated = true;
            }
            _ => unreachable!("admissibility and advancement must stay exhaustive"),
        }
        Ok(())
    }

    fn pop_hole(&mut self) -> Hole {
        self.holes
            .pop()
            .expect("an expression cue always owns one hole")
    }

    fn replace_current_hole<const N: usize>(
        &mut self,
        children: [(TypedValue, TypedHoleRole); N],
    ) -> Result<(), TypedHoleError> {
        let parent = *self
            .holes
            .last()
            .expect("an expression cue always owns one hole");
        let mut replacements = [parent; N];
        for (index, (expected, role)) in children.into_iter().enumerate() {
            let child_ordinal = u8::try_from(index).expect("typed constructors have few children");
            let address = parent.address.child(child_ordinal).ok_or(
                TypedHoleError::StructuralAddressOverflow {
                    address: parent.address,
                    child_ordinal,
                },
            )?;
            replacements[index] = Hole {
                expected,
                variable_zero: parent.variable_zero,
                role,
                address,
            };
        }
        self.pop_hole();
        self.holes.extend(replacements.into_iter().rev());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SurfaceKind as S, TypedHoleAddress, TypedHoleCue, TypedHoleError, TypedHoleRole,
        TypedHoleState, TypedValue, MAX_HOLE_ADDRESS_DEPTH,
    };

    fn advance_stream(state: &mut TypedHoleState, stream: &[S]) {
        for &surface in stream {
            state.advance(surface).unwrap();
        }
    }

    fn address(ordinals: &[u8]) -> TypedHoleAddress {
        ordinals
            .iter()
            .fold(TypedHoleAddress::root(), |path, &ordinal| {
                path.child(ordinal).unwrap()
            })
    }

    #[test]
    fn scalar_add_advances_through_literal_and_ordinary_eos() {
        let mut state = TypedHoleState::unary(TypedValue::Int, TypedValue::Int).unwrap();
        advance_stream(
            &mut state,
            &[
                S::Add,
                S::Variable0,
                S::Integer,
                S::Plus,
                S::Digit(5),
                S::End,
                S::Eos,
            ],
        );
        assert_eq!(state.cue(), TypedHoleCue::Terminated);
    }

    #[test]
    fn map_absolute_advances_function_body_collection_and_eos() {
        let mut state = TypedHoleState::unary(TypedValue::ListInt, TypedValue::ListInt).unwrap();
        advance_stream(
            &mut state,
            &[
                S::Map,
                S::Bind,
                S::Absolute,
                S::Variable0,
                S::Variable0,
                S::Eos,
            ],
        );
        assert_eq!(state.cue(), TypedHoleCue::Terminated);
    }

    #[test]
    fn map_add_advances_nested_literal_then_returns_to_list_hole() {
        let mut state = TypedHoleState::unary(TypedValue::ListInt, TypedValue::ListInt).unwrap();
        advance_stream(
            &mut state,
            &[
                S::Map,
                S::Bind,
                S::Add,
                S::Variable0,
                S::Integer,
                S::Plus,
                S::Digit(3),
                S::End,
            ],
        );
        assert_eq!(state.cue(), TypedHoleCue::Expression(TypedValue::ListInt));
        advance_stream(&mut state, &[S::Variable0, S::Eos]);
        assert_eq!(state.cue(), TypedHoleCue::Terminated);
    }

    #[test]
    fn root_and_map_body_int_holes_have_the_same_transferable_cue() {
        let root = TypedHoleState::unary(TypedValue::Int, TypedValue::Int).unwrap();
        let mut nested = TypedHoleState::unary(TypedValue::ListInt, TypedValue::ListInt).unwrap();
        advance_stream(&mut nested, &[S::Map, S::Bind]);

        assert_eq!(root.cue(), TypedHoleCue::Expression(TypedValue::Int));
        assert_eq!(nested.cue(), root.cue());
        assert!(root.admits(S::Add));
        assert!(nested.admits(S::Add));
        assert!(!nested.admits(S::Map));
    }

    #[test]
    fn invalid_surface_and_noncanonical_literal_stop_without_mutating_state() {
        let mut state = TypedHoleState::unary(TypedValue::Int, TypedValue::Int).unwrap();
        let before = state.clone();
        assert_eq!(
            state.advance(S::Map),
            Err(TypedHoleError::Inadmissible {
                cue: TypedHoleCue::Expression(TypedValue::Int),
                surface: S::Map,
            })
        );
        assert_eq!(state, before);

        advance_stream(&mut state, &[S::Integer, S::Minus]);
        let before_zero = state.clone();
        assert!(state.advance(S::Digit(0)).is_err());
        assert_eq!(state, before_zero);
        assert!(state.advance(S::Eos).is_err());

        let mut deep = TypedHoleState::unary(TypedValue::Int, TypedValue::Int).unwrap();
        for _ in 0..MAX_HOLE_ADDRESS_DEPTH {
            deep.advance(S::Absolute).unwrap();
        }
        let before_overflow = deep.clone();
        assert_eq!(
            deep.advance(S::Absolute),
            Err(TypedHoleError::StructuralAddressOverflow {
                address: before_overflow.address(),
                child_ordinal: 0,
            })
        );
        assert_eq!(deep, before_overflow);
    }

    #[test]
    fn nested_clamp_trace_preserves_types_roles_and_structural_addresses() {
        let mut state = TypedHoleState::unary(TypedValue::Int, TypedValue::Int).unwrap();
        let stream = [
            (
                S::If,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Head,
                &[][..],
            ),
            (
                S::LessThan,
                TypedHoleCue::Expression(TypedValue::Bool),
                TypedHoleRole::Child0,
                &[0][..],
            ),
            (
                S::Variable0,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child0,
                &[0, 0][..],
            ),
            (
                S::Integer,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child1,
                &[0, 1][..],
            ),
            (
                S::Minus,
                TypedHoleCue::LiteralSign,
                TypedHoleRole::Child1,
                &[0, 1][..],
            ),
            (
                S::Digit(5),
                TypedHoleCue::LiteralFirstNonzeroDigit,
                TypedHoleRole::Child1,
                &[0, 1][..],
            ),
            (
                S::End,
                TypedHoleCue::LiteralDigitOrEnd,
                TypedHoleRole::Child1,
                &[0, 1][..],
            ),
            (
                S::Integer,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child1,
                &[1][..],
            ),
            (
                S::Minus,
                TypedHoleCue::LiteralSign,
                TypedHoleRole::Child1,
                &[1][..],
            ),
            (
                S::Digit(5),
                TypedHoleCue::LiteralFirstNonzeroDigit,
                TypedHoleRole::Child1,
                &[1][..],
            ),
            (
                S::End,
                TypedHoleCue::LiteralDigitOrEnd,
                TypedHoleRole::Child1,
                &[1][..],
            ),
            (
                S::If,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child2,
                &[2][..],
            ),
            (
                S::LessThan,
                TypedHoleCue::Expression(TypedValue::Bool),
                TypedHoleRole::Child0,
                &[2, 0][..],
            ),
            (
                S::Integer,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child0,
                &[2, 0, 0][..],
            ),
            (
                S::Plus,
                TypedHoleCue::LiteralSign,
                TypedHoleRole::Child0,
                &[2, 0, 0][..],
            ),
            (
                S::Digit(5),
                TypedHoleCue::LiteralFirstDigit,
                TypedHoleRole::Child0,
                &[2, 0, 0][..],
            ),
            (
                S::End,
                TypedHoleCue::LiteralDigitOrEnd,
                TypedHoleRole::Child0,
                &[2, 0, 0][..],
            ),
            (
                S::Variable0,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child1,
                &[2, 0, 1][..],
            ),
            (
                S::Integer,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child1,
                &[2, 1][..],
            ),
            (
                S::Plus,
                TypedHoleCue::LiteralSign,
                TypedHoleRole::Child1,
                &[2, 1][..],
            ),
            (
                S::Digit(5),
                TypedHoleCue::LiteralFirstDigit,
                TypedHoleRole::Child1,
                &[2, 1][..],
            ),
            (
                S::End,
                TypedHoleCue::LiteralDigitOrEnd,
                TypedHoleRole::Child1,
                &[2, 1][..],
            ),
            (
                S::Variable0,
                TypedHoleCue::Expression(TypedValue::Int),
                TypedHoleRole::Child2,
                &[2, 2][..],
            ),
            (S::Eos, TypedHoleCue::Eos, TypedHoleRole::Head, &[][..]),
        ];

        for (surface, cue, role, ordinals) in stream {
            assert_eq!(state.cue(), cue);
            assert_eq!(state.role(), role);
            assert_eq!(state.address(), address(ordinals));
            assert_eq!(state.address().depth(), ordinals.len() as u8);
            for (index, &ordinal) in ordinals.iter().enumerate() {
                assert_eq!(state.address().child_ordinal(index as u8), Some(ordinal));
            }
            assert!(state.admits(surface));
            state.advance(surface).unwrap();
        }

        assert_eq!(state.cue(), TypedHoleCue::Terminated);
        assert_ne!(address(&[0, 0]), address(&[2, 0, 0]));
        assert_ne!(address(&[0, 1]), address(&[2, 0, 1]));
        assert_ne!(address(&[1]), address(&[2, 1]));
        assert_eq!(address(&[2, 1]).packed_ordinals(), 6);
    }
}
