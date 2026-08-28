//! Minimal structural state for native typed-program holes.
//!
//! This module knows only the small typed surface language. It has no prompt,
//! task-family, token-ID, candidate-list, or temporal-runtime dependency. The
//! cue exposed to a future physical workspace is therefore transferable: an
//! `Int` hole at the root and an `Int` hole inside `map` have the same identity.

const MAX_LITERAL_DIGITS: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TypedValue {
    Int,
    ListInt,
    IntToInt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SurfaceKind {
    Add,
    Absolute,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Hole {
    expected: TypedValue,
    variable_zero: TypedValue,
    role: TypedHoleRole,
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
            }],
            literal: None,
            literal_role: TypedHoleRole::Head,
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

    pub(crate) fn admits(&self, surface: SurfaceKind) -> bool {
        match (self.cue(), surface) {
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Add)
            | (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Absolute)
            | (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Integer)
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
                let hole = self.pop_hole();
                self.holes.push(Hole {
                    expected: TypedValue::Int,
                    variable_zero: hole.variable_zero,
                    role: TypedHoleRole::Child1,
                });
                self.holes.push(Hole {
                    expected: TypedValue::Int,
                    variable_zero: hole.variable_zero,
                    role: TypedHoleRole::Child0,
                });
            }
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Absolute) => {
                let hole = self.pop_hole();
                self.holes.push(Hole {
                    expected: TypedValue::Int,
                    variable_zero: hole.variable_zero,
                    role: TypedHoleRole::Child0,
                });
            }
            (TypedHoleCue::Expression(TypedValue::ListInt), SurfaceKind::Map) => {
                let hole = self.pop_hole();
                self.holes.push(Hole {
                    expected: TypedValue::ListInt,
                    variable_zero: hole.variable_zero,
                    role: TypedHoleRole::Child1,
                });
                self.holes.push(Hole {
                    expected: TypedValue::IntToInt,
                    variable_zero: hole.variable_zero,
                    role: TypedHoleRole::Child0,
                });
            }
            (TypedHoleCue::Expression(TypedValue::IntToInt), SurfaceKind::Bind) => {
                self.pop_hole();
                self.holes.push(Hole {
                    expected: TypedValue::Int,
                    variable_zero: TypedValue::Int,
                    role: TypedHoleRole::Head,
                });
            }
            (TypedHoleCue::Expression(_), SurfaceKind::Variable0) => {
                self.pop_hole();
            }
            (TypedHoleCue::Expression(TypedValue::Int), SurfaceKind::Integer) => {
                self.literal_role = self.pop_hole().role;
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
}

#[cfg(test)]
mod tests {
    use super::{SurfaceKind as S, TypedHoleCue, TypedHoleError, TypedHoleState, TypedValue};

    fn advance_stream(state: &mut TypedHoleState, stream: &[S]) {
        for &surface in stream {
            state.advance(surface).unwrap();
        }
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
    }
}
