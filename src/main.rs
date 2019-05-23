use bzip2::bufread::BzDecoder;
use detector_char_classes::*;
use detone::IterDecomposeVietnamese;
use encoding_rs::Encoding;
use encoding_rs::WINDOWS_1251;

use encoding_rs::WINDOWS_1250;
use encoding_rs::WINDOWS_1250_INIT;
use encoding_rs::WINDOWS_1251_INIT;
use encoding_rs::WINDOWS_1252_INIT;
use encoding_rs::WINDOWS_1253_INIT;
use encoding_rs::WINDOWS_1254;
use encoding_rs::WINDOWS_1254_INIT;
use encoding_rs::WINDOWS_1255_INIT;
use encoding_rs::WINDOWS_1256;
use encoding_rs::WINDOWS_1256_INIT;
use encoding_rs::WINDOWS_1257_INIT;
use encoding_rs::WINDOWS_1258;
use encoding_rs::WINDOWS_1258_INIT;
use encoding_rs::WINDOWS_874_INIT;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use unic_normal::StrNormalForm;
// use utf8reader::UTF8Reader;
use unicode_reader::CodePoints;

struct CharMap {
    arr: [u8; 3674],
}

impl CharMap {
    fn new(char_classes: &'static [&'static [char]], windows_encoding: &'static Encoding) -> Self {
        let mut ret = CharMap { arr: [0u8; 3674] };
        for (i, chars) in char_classes.iter().enumerate() {
            let class = i as u8;
            for &c in chars.iter() {
                ret.arr[c as usize] = class;

                let upper = if windows_encoding == WINDOWS_1254 && c == 'i' {
                    'İ'
                } else if c == 'ς' {
                    // Intentionally not handling final sigma to match
                    // detection-time mapping.
                    continue;
                } else {
                    let mut iter = c.to_uppercase();
                    let first = iter.next().unwrap();
                    if let Some(_) = iter.next() {
                        continue;
                    }
                    first
                };
                ret.arr[upper as usize] = class;
            }
        }
        if !is_latin(windows_encoding) {
            for c in b'a'..=b'z' {
                ret.arr[c as usize] = 0xFE;
            }
            for c in b'A'..=b'Z' {
                ret.arr[c as usize] = 0xFE;
            }
            if windows_encoding == WINDOWS_1256 {
                for &c in ARABIC_FRENCH.iter() {
                    ret.arr[c as usize] = 0xFE;
                }
            }
        }
        ret
    }

    fn get(&self, c: char) -> u8 {
        let s = c as usize;
        if s < self.arr.len() {
            return self.arr[s];
        }
        0u8
    }
}

#[derive(Debug)]
struct EncodingClass {
    char_classes: &'static [&'static [char]],
    encodings: &'static [&'static Encoding],
    languages: &'static [&'static str],
    name: &'static str,
}

impl EncodingClass {
    fn train(&'static self, dir: &Path) -> (Vec<f64>, &'static Self) {
        let windows_encoding = self.encodings[0];

        let map = CharMap::new(self.char_classes, windows_encoding);

        let (ascii_classes, non_ascii_classes) = count_ascii_classes(self.char_classes);

        let mut language_scores = Vec::with_capacity(self.languages.len());
        for lang in self.languages {
            eprintln!("Training {:?}", lang);
            let corpus = find_file(dir, lang);
            let mut scores = train_one(
                &corpus,
                &map,
                ascii_classes,
                non_ascii_classes,
                windows_encoding,
                false,
            );
            divide_by_class_size(
                &mut scores,
                self.char_classes,
                ascii_classes,
                non_ascii_classes,
            );

	    let mut max = 0.0f64;
        for &score in scores.iter() {
            if score > max {
                max = score;
            }
        }

    	println!("MAX {}: {:?}", lang, max);

            language_scores.push(scores);
            if windows_encoding == WINDOWS_1258 {
                let mut scores = train_one(
                    &corpus,
                    &map,
                    ascii_classes,
                    non_ascii_classes,
                    windows_encoding,
                    true,
                );
                divide_by_class_size(
                    &mut scores,
                    self.char_classes,
                    ascii_classes,
                    non_ascii_classes,
                );

	    let mut max = 0.0f64;
        for &score in scores.iter() {
            if score > max {
                max = score;
            }
        }

    	println!("MAX {}: {:?}", lang, max);

                language_scores.push(scores);
            }
        }

        (merge(language_scores), self)
    }
}

static ENCODING_CLASSES: [EncodingClass; 10] = [
    EncodingClass {
        char_classes: &CENTRAL,
        encodings: &[&WINDOWS_1250_INIT],
        languages: &["pl", "hu", "sh", "cs", "ro", "sk", "hr", "sl", "bs", "sq"],
        name: "central",
    },
    EncodingClass {
        char_classes: &CYRILLIC,
        encodings: &[&WINDOWS_1251_INIT],
        // kk, tt, tg, and os don't fit
        // mn uses mapping to uk letters
        languages: &["ru", "uk", "sr", "bg", "ce", "be", "mk", "mn"],
        name: "cyrillic",
    },
    EncodingClass {
        char_classes: &WESTERN,
        encodings: &[&WINDOWS_1252_INIT],
        // Intentionally omitting ASCII languages like en, nl, id, so, sw, various Malay-alphabet languages
        // Should et and sq be also here?
        languages: &[
            "sv", "de", "fr", "it", "es", "pt", "ca", "no", "fi", "eu", "da", "et", "gl", "nn",
            "oc", "br", "lb", "ht", "ga", "is", "an", "wa", "gd", "fo", "li",
        ],
        name: "western",
    },
    EncodingClass {
        char_classes: &GREEK,
        encodings: &[&WINDOWS_1253_INIT],
        languages: &["el"],
        name: "greek",
    },
    EncodingClass {
        char_classes: &TURKISH,
        encodings: &[&WINDOWS_1254_INIT],
        languages: &["tr", "az", "ku"],
        name: "turkish",
    },
    EncodingClass {
        char_classes: &HEBREW,
        encodings: &[&WINDOWS_1255_INIT],
        languages: &["he", "yi"],
        name: "hebrew",
    },
    EncodingClass {
        char_classes: &ARABIC,
        encodings: &[&WINDOWS_1256_INIT],
        languages: &["ar", "fa", "ur"],
        name: "arabic",
    },
    EncodingClass {
        char_classes: &BALTIC,
        encodings: &[&WINDOWS_1257_INIT],
        languages: &["lt", "et", "lv"],
        name: "baltic",
    },
    EncodingClass {
        char_classes: &VIETNAMESE,
        encodings: &[&WINDOWS_1258_INIT],
        languages: &["vi"],
        name: "vietnamese",
    },
    EncodingClass {
        char_classes: &THAI,
        encodings: &[&WINDOWS_874_INIT],
        languages: &["th"],
        name: "thai",
    },
];

fn find_file(dir: &Path, lang: &str) -> PathBuf {
    for entry in dir
        .read_dir()
        .expect("Reading the corpus directory failed.")
    {
        if let Ok(entry) = entry {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(lang) {
                return entry.path();
            }
        }
    }
    eprintln!("Error: No corpus for: {}", lang);
    std::process::exit(-4);
}

fn count_ascii_classes(char_classes: &'static [&'static [char]]) -> (usize, usize) {
    let ascii_classes = char_classes
        .iter()
        .map(|c| c[0])
        .take_while(|c| *c < '\u{80}')
        .count();
    if ascii_classes == 1 {
        (0, char_classes.len())
    } else {
        (ascii_classes, char_classes.len() - ascii_classes)
    }
}

fn open_bzip2(path: &Path) -> impl Iterator<Item = char> {
    let dec = BzDecoder::new(BufReader::new(File::open(path).unwrap()));
    CodePoints::from(BufReader::new(dec))
        .map(|r| r.unwrap())
        .take(50000) // XXX remove
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
    if x == 254 || y == 254 {
        return None;
    }
    if x < ascii_classes && y < ascii_classes {
        return None;
    }
    if y >= ascii_classes {
        return Some(
            (ascii_classes * non_ascii_classes)
                + (ascii_classes + non_ascii_classes) * (y - ascii_classes)
                + x,
        );
    }
    Some(y * non_ascii_classes + x - ascii_classes)
}

fn compute_scores<I: Iterator<Item = char>>(
    iter: I,
    classes: &CharMap,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Vec<f64> {
    let score_len = non_ascii_classes * non_ascii_classes + 2 * (ascii_classes * non_ascii_classes);
    let mut scores = Vec::new();
    scores.resize(score_len, 0u64);

    let mut total = 0u64;

    let mut prev = 0usize;

    for c in iter {
        let current = classes.get(c) as usize;
        if let Some(index) = compute_index(prev, current, ascii_classes, non_ascii_classes) {
            scores[index] += 1;
            total += 1;
        }
        prev = current;
    }

    if ascii_classes == 0 {
        assert_eq!(scores[0], 0, "Space doesn't pair with itself.");
    }

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
    classes: &CharMap,
    ascii_classes: usize,
    non_ascii_classes: usize,
    encoding: &'static Encoding,
    orthographic_vietnamese: bool,
) -> Vec<f64> {
    let iter = open_bzip2(path);

    if encoding == WINDOWS_1256 {
        // Map non-ASCII Latin to ASCII Latin
        compute_scores(
            iter.nfc().map(|c| {
                // XXX upper case
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
            iter.nfc()
                .decompose_vietnamese_tones(orthographic_vietnamese),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1254 {
        // Map Azeri ə to ä.
        compute_scores(
            iter.nfc().map(|c| match c {
                'Ə' => 'Ä',
                'ə' => 'ä',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1250 {
        // Map Romanian comma-below characters to cedilla versions,
        // because legacy encodings (other than ISO-8859-16, which we
        // won't detect) only have the latter.
        compute_scores(
            iter.nfc().map(|c| match c {
                'ț' => 'ţ',
                'ș' => 'ş',
                'Ț' => 'Ţ',
                'Ș' => 'Ş',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1251 {
        // Map Mongolian characters to Ukranian substitutes
        compute_scores(
            iter.nfc().map(|c| match c {
                'Ү' => 'Ї',
                'ү' => 'ї',
                'Ө' => 'Є',
                'ө' => 'є',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else {
        compute_scores(iter.nfc(), classes, ascii_classes, non_ascii_classes)
    }
}

fn further_than_epsilon(a: u8, b: u8, e: u8) -> bool {
	let delta = if a < b {
		b - a
	} else {
		a - b
	};
	delta > e
}

fn suggest_merges(scores: &Vec<u8>, encoding_class: &'static EncodingClass) {
    let (ascii_classes, non_ascii_classes) = count_ascii_classes(encoding_class.char_classes);
    for e in 0..2 {
    	println!("Epsilon = {:?} ---------------------------------", e);
        for i in 0..encoding_class.char_classes.len() {
            'mid: for j in 0..i {
            	if i >= ascii_classes && j < ascii_classes || j >= ascii_classes && i < ascii_classes {
            		// Don't suggest merging ASCII and non-ASCII
            		continue;
            	}
                for k in 0..encoding_class.char_classes.len() {
                    let i_index_1 = compute_index(i, k, ascii_classes, non_ascii_classes);
                    let j_index_1 = compute_index(j, k, ascii_classes, non_ascii_classes);
                    if let (Some(i_index), Some(j_index)) = (i_index_1, j_index_1) {
                        if further_than_epsilon(scores[i_index], scores[j_index], e) {
                            continue 'mid;
                        }
                    }
                    let i_index_2 = compute_index(k, i, ascii_classes, non_ascii_classes);
                    let j_index_2 = compute_index(k, j, ascii_classes, non_ascii_classes);
                    if let (Some(i_index), Some(j_index)) = (i_index_2, j_index_2) {
                        if further_than_epsilon(scores[i_index], scores[j_index], e) {
                            continue 'mid;
                        }
                    }
                }
                // Suggest
                println!("Epsilon: {:?}, {}, merge: {:?} & {:?}", e, encoding_class.name, encoding_class.char_classes[i], encoding_class.char_classes[j]);
            }
        }
    }
}

fn train_with_dir(dir: &Path) {
    let mut float_scores = Vec::with_capacity(ENCODING_CLASSES.len());
    for class in ENCODING_CLASSES.iter() {
        float_scores.push(class.train(dir));
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
    for (vec, encoding_class) in float_scores.iter() {
        let mut byte_vec = Vec::new();
        byte_vec.resize(vec.len(), 0u8);
        for (b, f) in byte_vec.iter_mut().zip(vec.into_iter()) {
            *b = f64::floor((f / max) * 255.5) as u8;
        }
        scores.push((byte_vec, encoding_class));
    }

    for (float_vec, encoding_class) in float_scores.iter() {
	    let mut max = 0.0f64;
        for &score in float_vec.iter() {
            if score > max {
                max = score;
            }
        }

    	println!("MAX {}: {:?}", encoding_class.name, max);
	}

    for (byte_vec, encoding_class) in scores.iter() {
		println!("MAX byte {}: {:?}", encoding_class.name, byte_vec.iter().max());
    }

    for (byte_vec, encoding_class) in scores.iter() {
        suggest_merges(&byte_vec, encoding_class);
    }
}

fn download_corpus(dir: &Path) {
    let prefix = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/";
    let date = "20190420";
    let mut curl = Command::new("curl");
    curl.current_dir(dir);
    curl.arg("--remote-name-all");
    for encoding_class in ENCODING_CLASSES.iter() {
        for lang in encoding_class.languages.iter() {
            let mut url = String::new();
            url.push_str(prefix);
            url.push_str(lang);
            url.push_str("wiki/");
            url.push_str(date);
            url.push_str("/");
            url.push_str(lang);
            url.push_str("wiki-");
            url.push_str(date);
            url.push_str("-pages-articles.xml.bz2");
            curl.arg(url);
        }
    }
    curl.output().expect("Executing curl failed");
}

fn main() {
    let mut args = std::env::args_os();
    if args.next().is_none() {
        eprintln!("Error: Program name missing from arguments.");
        std::process::exit(-1);
    }
    if let Some(path) = args.next() {
        if let Some(download) = args.next() {
            if "--download" == download {
                download_corpus(Path::new(&path));
            }
        } else {
            train_with_dir(Path::new(&path));
        }
    } else {
        eprintln!("Error: Too few arguments.");
        std::process::exit(-2);
    };
}

#[cfg(test)]
mod tests {
    use super::compute_index;
    #[test]
    fn test_compute_index() {
        //    0  1  2  3  4
        //  +--+--+--+--+--+
        // 0|  |  | 0| 1| 2|
        //  +--+--+--+--+--+
        // 1|  |  | 3| 4| 5|
        //  +--+--+--+--+--+
        // 2| 6| 7| 8| 9|10|
        //  +--+--+--+--+--+
        // 3|11|12|13|14|15|
        //  +--+--+--+--+--+
        // 4|16|17|18|19|20|
        //  +--+--+--+--+--+
        assert_eq!(compute_index(1, 1, 2, 3), None);
        assert_eq!(compute_index(2, 0, 2, 3), Some(0));
        assert_eq!(compute_index(4, 0, 2, 3), Some(2));
        assert_eq!(compute_index(2, 1, 2, 3), Some(3));
        assert_eq!(compute_index(4, 1, 2, 3), Some(5));
        assert_eq!(compute_index(0, 2, 2, 3), Some(6));
        assert_eq!(compute_index(4, 2, 2, 3), Some(10));
        assert_eq!(compute_index(0, 3, 2, 3), Some(11));
        assert_eq!(compute_index(4, 3, 2, 3), Some(15));
        assert_eq!(compute_index(0, 4, 2, 3), Some(16));
        assert_eq!(compute_index(4, 4, 2, 3), Some(20));
    }
}
