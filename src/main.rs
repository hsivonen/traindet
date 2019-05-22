use bzip2::bufread::BzDecoder;
use detector_char_classes::*;
use detone::IterDecomposeVietnamese;
use encoding_rs::Encoding;
use encoding_rs::WINDOWS_1254;
use encoding_rs::WINDOWS_1256;
use encoding_rs::WINDOWS_1258;
use encoding_rs::WINDOWS_1258_INIT;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;
use unic_normal::StrNormalForm;
use unicode_reader::CodePoints;

#[derive(Debug)]
struct EncodingClass {
    char_classes: &'static [&'static [char]],
    encodings: &'static [&'static Encoding],
    languages: &'static [&'static str],
    name: &'static str,
}

impl EncodingClass {
    fn train(&'static self) -> (Vec<f64>, &'static Self) {
        let mut map = HashMap::new();
        for (i, chars) in self.char_classes.iter().enumerate() {
            let class = i as u8;
            for &c in chars.iter() {
                map.insert(c, class);
            }
        }

        let windows_encoding = self.encodings[0];
        let (ascii_classes, non_ascii_classes) = count_ascii_classes(self.char_classes);

        let mut language_scores = Vec::with_capacity(self.languages.len());
        for lang in self.languages {
        	eprintln!("Training {:?}", lang);
            let mut scores = train_one(Path::new(lang), &map, ascii_classes, non_ascii_classes, windows_encoding, false);
            divide_by_class_size(&mut scores, self.char_classes, ascii_classes, non_ascii_classes);
            language_scores.push(scores);
            if windows_encoding == WINDOWS_1258 {
                let mut scores = train_one(Path::new(lang), &map, ascii_classes, non_ascii_classes, windows_encoding, true);
                divide_by_class_size(&mut scores, self.char_classes, ascii_classes, non_ascii_classes);
                language_scores.push(scores);
            }
        }

        (merge(language_scores), self)
    }
}

static ENCODING_CLASSES: [EncodingClass; 1] = [EncodingClass {
    char_classes: &VIETNAMESE,
    encodings: &[&WINDOWS_1258_INIT],
    languages: &["vi"],
    name: "vietnamese",
}];

fn count_ascii_classes(char_classes: &'static [&'static [char]]) -> (usize, usize) {
	let ascii_classes = char_classes.iter().map(|c| c[0]).take_while(|c| *c < '\u{80}').count();
	if ascii_classes == 1 {
		(0, char_classes.len())
	} else {
		(ascii_classes, char_classes.len() - ascii_classes)
	}
}

fn open_bzip2(path: &Path) -> impl Iterator<Item = char> {
    let dec = BzDecoder::new(BufReader::new(File::open(path).unwrap()));
    CodePoints::from(dec.bytes()).map(|r| r.unwrap())
}

fn merge(language_scores: Vec<Vec<f64>>) -> Vec<f64> {
	let mut iter = language_scores.into_iter();
	let mut ret = iter.next().unwrap();
	for vec in iter {
		assert_eq!(ret.len(), vec.len());
		for (r, v) in ret.iter_mut().zip(vec.into_iter()) {
			*r = f64::max(*r, v);
		}
	}
	ret
}

fn divide_by_class_size(
    scores: &mut Vec<f64>,
    classes: &'static [&'static [char]],
    ascii_classes: usize,
    non_ascii_classes: usize,
) {
    for (i, size) in classes.iter().map(|c| c.len()).enumerate() {
        if size > 1 {
            let divisor = size as f64;
            for j in 0..classes.len() {
                if let Some(index) = compute_index(i, j, ascii_classes, non_ascii_classes) {
                    scores[index] /= divisor;
                }
                if let Some(index) = compute_index(j, i, ascii_classes, non_ascii_classes) {
                    scores[index] /= divisor;
                }
            }
        }
    }
}

#[inline(always)]
fn compute_index(
    x: usize,
    y: usize,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Option<usize> {
    if x == 0 && y == 0 {
        return None;
    }
    if ascii_classes == 0 {
        return Some(x * y);
    }
    if x < ascii_classes && y < ascii_classes {
        return None;
    }
    if y >= ascii_classes {
        Some(x * y - ascii_classes * ascii_classes);
    }
    Some(y * non_ascii_classes + x - ascii_classes)
}

fn compute_scores<I: Iterator<Item = char>>(
    iter: I,
    classes: &HashMap<char, u8>,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Vec<f64> {
    let score_len = non_ascii_classes * non_ascii_classes + 2 * (ascii_classes * non_ascii_classes);
    let mut scores = Vec::new();
    scores.resize(score_len, 0u64);

    let mut total = 0u64;

    let mut prev = Some(0usize);

    for c in iter {
        let current = if ascii_classes == 0 && c >= 'a' && c <= 'z' {
            // For non-Latin, ASCII doesn't pair with anything. By now,
            // windows-1256 accented French characters have been mapped
            // to `'a'` to be caught here.
            None
        } else {
            Some(*classes.get(&c).unwrap_or(&0u8) as usize)
        };
        if let (Some(prev_unwrap), Some(current_unwrap)) = (prev, current) {
            if let Some(index) = compute_index(
                prev_unwrap,
                current_unwrap,
                ascii_classes,
                non_ascii_classes,
            ) {
                scores[index] += 1;
                total += 1;
            }
        }
        prev = current;
    }

    assert_eq!(scores[0], 0, "Space doesn't pair with itself.");

    let mut float_scores = Vec::with_capacity(score_len);
    let float_total = total as f64;
    for score in scores {
        let float_score = score as f64;
        float_scores.push(float_score / float_total);
    }

    float_scores
}

fn train_one(
    path: &Path,
    classes: &HashMap<char, u8>,
    ascii_classes: usize,
    non_ascii_classes: usize,
    encoding: &'static Encoding,
    orthographic_vietnamese: bool,
) -> Vec<f64> {
    let iter = open_bzip2(path);

    if encoding == WINDOWS_1256 {
        // Map non-ASCII Latin to ASCII Latin
        compute_scores(
            iter.flat_map(|c| c.to_lowercase()).nfc().map(|c| {
                if ARABIC_FRENCH.iter().find(|&&x| x == c).is_some() {
                    'a'
                } else {
                    c
                }
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1258 {
        // Decompose tones
        compute_scores(
            iter.flat_map(|c| c.to_lowercase())
                .nfc()
                .decompose_vietnamese_tones(orthographic_vietnamese),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1254 {
        // Perform special I casing and map Azeri ə to ä.
        compute_scores(
            iter.map(|c| if c == 'I' { 'ı' } else { c })
                .flat_map(|c| c.to_lowercase())
                .nfc()
                .map(|c| if c == 'ə' { 'ä' } else { c }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else {
        // Intentionally not handling final sigma to match
        // detection-time mapping.
        // Map Romanian comma-below characters to cedilla versions,
        // because legacy encodings (other than ISO-8859-16, which we
        // won't detect) only have the latter.
        compute_scores(
            iter.flat_map(|c| c.to_lowercase()).nfc().map(|c| {
                if c == 'ț' {
                    'ţ'
                } else if c == 'ș' {
                    'ş'
                } else {
                    c
                }
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    }
}

fn main() {
	let mut float_scores = Vec::with_capacity(ENCODING_CLASSES.len());
    for class in ENCODING_CLASSES.iter() {
        float_scores.push(class.train());
    }
    let mut max = 0.0f64;
    for (vec, _) in float_scores.iter() {
    	for &score in vec.iter() {
    		if score > max {
    			max = score;
    		}
    	}
    }
    let mut scores = Vec::with_capacity(float_scores.len());
    for (vec, encoding_class) in float_scores {
    	let mut byte_vec = Vec::new();
    	byte_vec.resize(vec.len(), 0u8);
    	for (b, f) in byte_vec.iter_mut().zip(vec.into_iter()) {
    		*b = f64::floor((f / max) * 255.5) as u8;
    	}
    	scores.push((byte_vec, encoding_class));
    }
    
}
