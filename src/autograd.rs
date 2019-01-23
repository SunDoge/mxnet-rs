use mxnet_sys::{
    MXAutogradIsRecording, MXAutogradIsTraining, MXAutogradSetIsRecording, MXAutogradSetIsTraining,
};

pub struct RecordingStateScope {
    enter_is_record: Option<bool>,
    enter_train_mode: Option<bool>,
    prev_is_record: Option<bool>,
    prev_train_mode: Option<bool>,
}

impl RecordingStateScope {
    pub fn new(is_record: Option<bool>, train_mode: Option<bool>) -> RecordingStateScope {
        let mut state = RecordingStateScope {
            enter_is_record: is_record,
            enter_train_mode: train_mode,
            prev_is_record: None,
            prev_train_mode: None,
        };

        state.enter();
        state
    }

    pub fn enter(&mut self) {
        if let Some(enter_is_record) = self.enter_is_record {
            self.prev_is_record = Some(set_recording(enter_is_record));
        }

        if let Some(enter_train_mode) = self.enter_train_mode {
            self.prev_train_mode = Some(set_training(enter_train_mode));
        }
    }
}

impl Drop for RecordingStateScope {
    fn drop(&mut self) {
        if self.enter_is_record.is_some() && self.prev_is_record != self.enter_is_record {
            set_recording(self.prev_is_record.unwrap());
        }

        if self.enter_train_mode.is_some() && self.prev_train_mode != self.enter_train_mode {
            set_training(self.prev_train_mode.unwrap());
        }
    }
}

fn set_recording(is_recording: bool) -> bool {
    let mut prev = 0;
    check_call!(MXAutogradSetIsRecording(is_recording as i32, &mut prev));
    prev != 0
}

fn set_training(train_mode: bool) -> bool {
    let mut prev = 0;
    check_call!(MXAutogradSetIsTraining(train_mode as i32, &mut prev));
    prev != 0
}

fn is_recording() -> bool {
    let mut curr = false;
    check_call!(MXAutogradIsRecording(&mut curr));
    curr
}

fn is_training() -> bool {
    let mut curr = false;
    check_call!(MXAutogradIsTraining(&mut curr));
    curr
}

/// Returns an autograd recording scope context to be used in 'with' statement
/// and captures code that needs gradients to be calculated.
pub fn record() -> RecordingStateScope {
    RecordingStateScope::new(Some(true), Some(true))
}

/// Returns a scope context to be used in 'with' statement for codes that do not need
/// gradients to be calculated.
pub fn pause() -> RecordingStateScope {
    RecordingStateScope::new(Some(false), Some(false))
}

pub fn train_mode() -> RecordingStateScope {
    RecordingStateScope::new(None, Some(true))
}

pub fn predict_mode() -> RecordingStateScope {
    RecordingStateScope::new(None, Some(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndarray;

    #[test]
    fn enter_and_exit() {
        let mut x = ndarray::NDArray::builder().data(&[1.0]).create();
        {
            let _ = record();

            x += 1.0;
            
        }

        println!("{}", x);
        
    }
}
