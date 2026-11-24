//! The original C LibAtrac9 behind a small stdin/stdout protocol (`tests/reference`).

use std::{
    ffi::OsString,
    fs,
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
    sync::OnceLock,
    time::Duration,
};

static EXECUTABLE: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug)]
pub struct Frame {
    pub status: u32,
    pub bytes_used: usize,
    pub samples: Vec<i16>,
    pub float_samples: Vec<f32>,
}

/// The reference executable: `ATRAC9_REFERENCE` if set, otherwise built once per
/// test binary with the system C compiler (`CC`, else `cc`, or `cl` on Windows).
pub fn executable() -> &'static PathBuf {
    EXECUTABLE
        .get_or_init(|| std::env::var_os("ATRAC9_REFERENCE").map_or_else(build, PathBuf::from))
}

fn build() -> PathBuf {
    let sources = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference");
    let mut files: Vec<PathBuf> = fs::read_dir(&sources)
        .expect("read tests/reference")
        .map(|entry| entry.expect("reference directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "c"))
        .collect();
    files.sort();
    // Each test binary builds its own copy, so binaries running in parallel never
    // overwrite an executable another one is using.
    let binary = std::env::current_exe().expect("test executable path");
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("reference")
        .join(binary.file_stem().expect("test executable name"));
    fs::create_dir_all(&directory).expect("create reference build directory");
    let output = directory.join(format!("atrac9-reference{}", std::env::consts::EXE_SUFFIX));

    let compiler = std::env::var_os("CC")
        .unwrap_or_else(|| OsString::from(if cfg!(windows) { "cl" } else { "cc" }));
    let msvc = Path::new(&compiler)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| {
            stem.eq_ignore_ascii_case("cl") || stem.eq_ignore_ascii_case("clang-cl")
        });
    let mut command = Command::new(&compiler);
    if msvc {
        let mut objects = directory.clone().into_os_string();
        objects.push(std::path::MAIN_SEPARATOR_STR);
        let mut executable = OsString::from("/Fe");
        executable.push(&output);
        let mut object_directory = OsString::from("/Fo");
        object_directory.push(objects);
        command
            .args(["/nologo", "/O2", "/fp:precise", "/W0"])
            .arg(executable)
            .arg(object_directory)
            .args(&files);
    } else {
        // No FMA contraction: Rust never fuses, and the comparison is bit exact.
        command
            .args(["-O2", "-ffp-contract=off", "-w", "-o"])
            .arg(&output)
            .args(&files)
            .arg("-lm");
    }
    let result = command.output().unwrap_or_else(|error| {
        panic!(
            "could not run the C compiler `{}`: {error}. Install a C compiler, set CC, \
             or point ATRAC9_REFERENCE at a prebuilt reference (see tests/reference/README.md)",
            Path::new(&compiler).display()
        )
    });
    assert!(
        result.status.success(),
        "building the C reference failed:\n{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    output
}

fn request(config: [u8; 4], frames: &[Vec<u8>], repeats: u32) -> io::Result<Vec<u8>> {
    let mut child = Command::new(executable())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let input = child.stdin.as_mut().expect("piped stdin");
        input.write_all(b"AT9O")?;
        input.write_all(&config)?;
        input.write_all(&(frames.len() as u32).to_le_bytes())?;
        input.write_all(&repeats.to_le_bytes())?;
        for frame in frames {
            input.write_all(&(frame.len() as u32).to_le_bytes())?;
            input.write_all(frame)?;
        }
    }
    drop(child.stdin.take());
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "C reference exited {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    Ok(output.stdout)
}

fn read32(input: &mut &[u8]) -> io::Result<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub fn decode(config: [u8; 4], frames: &[Vec<u8>]) -> io::Result<Vec<Frame>> {
    read_frames(request(config, frames, 0)?, frames.len())
}

pub fn decode_superframes(config: [u8; 4], superframes: &[Vec<u8>]) -> io::Result<Vec<Frame>> {
    let count = superframes.len() * (1 << (config[3] >> 3 & 3));
    read_frames(request(config, superframes, u32::MAX)?, count)
}

#[allow(clippy::needless_pass_by_value)]
fn read_frames(response: Vec<u8>, count: usize) -> io::Result<Vec<Frame>> {
    let mut input = response.as_slice();
    assert_eq!(read32(&mut input)?, 0, "C initialization failed");
    let mut result = Vec::new();
    for _ in 0..count {
        let status = read32(&mut input)?;
        let bytes_used = read32(&mut input)? as usize;
        let count = read32(&mut input)? as usize;
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bytes = [0; 2];
            input.read_exact(&mut bytes)?;
            samples.push(i16::from_le_bytes(bytes));
        }
        let mut float_samples = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bytes = [0; 4];
            input.read_exact(&mut bytes)?;
            float_samples.push(f32::from_le_bytes(bytes));
        }
        result.push(Frame {
            status,
            bytes_used,
            samples,
            float_samples,
        });
    }
    assert_eq!(input, &[] as &[u8], "trailing C oracle protocol data");
    Ok(result)
}

pub fn timed(config: [u8; 4], frames: &[Vec<u8>], repeats: u32) -> io::Result<Duration> {
    let response = request(config, frames, repeats)?;
    let mut input = response.as_slice();
    assert_eq!(read32(&mut input)?, 0);
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes)?;
    assert_eq!(input, &[] as &[u8]);
    Ok(Duration::from_nanos(u64::from_le_bytes(bytes)))
}

/// Persistent dev-only C timing worker with one configuration and fixture batch.
#[derive(Debug)]
pub struct Timer {
    child: Child,
    input: Option<ChildStdin>,
    output: BufReader<ChildStdout>,
    errors: BufReader<ChildStderr>,
}

impl Timer {
    pub fn new(config: [u8; 4], frames: &[Vec<u8>]) -> io::Result<Self> {
        let mut child = Command::new(executable())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let input = child.stdin.take().expect("piped worker stdin");
        let output = BufReader::new(child.stdout.take().expect("piped worker stdout"));
        let errors = BufReader::new(child.stderr.take().expect("piped worker stderr"));
        let mut timer = Self {
            child,
            input: Some(input),
            output,
            errors,
        };
        {
            let input = timer.input.as_mut().expect("live worker input");
            input.write_all(b"AT9B")?;
            input.write_all(&config)?;
            input.write_all(&(frames.len() as u32).to_le_bytes())?;
            input.write_all(&0u32.to_le_bytes())?;
            for frame in frames {
                input.write_all(&(frame.len() as u32).to_le_bytes())?;
                input.write_all(frame)?;
            }
            input.flush()?;
        }
        let mut status = [0; 4];
        if let Err(error) = timer.output.read_exact(&mut status) {
            drop(timer.input.take());
            let exit = timer.child.wait()?;
            let mut details = String::new();
            timer.errors.read_to_string(&mut details)?;
            return Err(io::Error::other(format!(
                "C worker handshake failed: {error}; exit {exit}; {details}"
            )));
        }
        let status = u32::from_le_bytes(status);
        if status != 0 {
            return Err(io::Error::other(format!(
                "C worker initialization status {status}"
            )));
        }
        Ok(timer)
    }

    /// Measure a fresh-state batch; reset, IPC and table initialization are outside
    /// the C worker's timer.
    pub fn timed(&mut self, repeats: u32) -> io::Result<Duration> {
        if repeats == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero terminates the C timing worker",
            ));
        }
        let input = self.input.as_mut().expect("live worker input");
        input.write_all(&repeats.to_le_bytes())?;
        input.flush()?;
        let mut bytes = [0; 8];
        self.output.read_exact(&mut bytes)?;
        Ok(Duration::from_nanos(u64::from_le_bytes(bytes)))
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        // EOF terminates the waiting worker; no unconsumed response is pending.
        drop(self.input.take());
        let _ = self.child.wait();
    }
}
