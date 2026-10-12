// Adapted from https://github.com/PENGUINLIONG/graphi-t

// Copyright (c) 2019 Rendong Liang
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// JSON serialization/deserialization.
// @PENGUINLIONG
#include <sstream>
#include <charconv>
#include <cmath>
#include <locale>
// #include "gft/log.hpp"
#include "taichi/common/json.h"

namespace liong {
namespace json {

enum JsonTokenType {
  L_JSON_TOKEN_UNDEFINED,
  L_JSON_TOKEN_NULL,
  L_JSON_TOKEN_TRUE,
  L_JSON_TOKEN_FALSE,
  L_JSON_TOKEN_STRING,
  L_JSON_TOKEN_INT,
  L_JSON_TOKEN_FLOAT,
  L_JSON_TOKEN_COLON,
  L_JSON_TOKEN_COMMA,
  L_JSON_TOKEN_OPEN_BRACE,
  L_JSON_TOKEN_CLOSE_BRACE,
  L_JSON_TOKEN_OPEN_BRACKET,
  L_JSON_TOKEN_CLOSE_BRACKET,
};
struct JsonToken {
  JsonTokenType ty;
  int64_t num_int;
  double num_float;
  std::string str;
};

struct Tokenizer {
  const char *pos;
  const char *end;

  Tokenizer(const char *beg, const char *end) : pos(beg), end(end) {
  }

  // Check the range first before calling this method.
  bool unsafe_starts_with(const char *head) {
    auto i = 0;
    while (*head != '\0') {
      if (pos[i++] != *(head++)) {
        return false;
      }
    }
    return true;
  }
  bool next_token(JsonToken &out) {
    std::stringstream ss;
    while (pos != end) {
      char c = *pos;

      if (c == '\0') {
        break;
      }

      // Ignore whitespaces.
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        pos += 1;
        continue;
      }

      // Try parse scope punctuations.
      switch (c) {
        case ':':
          out.ty = L_JSON_TOKEN_COLON;
          pos += 1;
          return true;
        case ',':
          out.ty = L_JSON_TOKEN_COMMA;
          pos += 1;
          return true;
        case '{':
          out.ty = L_JSON_TOKEN_OPEN_BRACE;
          pos += 1;
          return true;
        case '}':
          out.ty = L_JSON_TOKEN_CLOSE_BRACE;
          pos += 1;
          return true;
        case '[':
          out.ty = L_JSON_TOKEN_OPEN_BRACKET;
          pos += 1;
          return true;
        case ']':
          out.ty = L_JSON_TOKEN_CLOSE_BRACKET;
          pos += 1;
          return true;
      }

      // Try parse numbers.
      if (c == '-' || (c >= '0' && c <= '9')) {
        const char *beg = pos;
        auto digit = [&] { return pos != end && *pos >= '0' && *pos <= '9'; };
        auto digits = [&] {
          if (!digit())
            throw JsonException("json number requires a digit");
          do {
            ++pos;
          } while (digit());
        };
        if (*pos == '-')
          ++pos;
        if (pos != end && *pos == '0')
          ++pos;
        else
          digits();
        out.ty = L_JSON_TOKEN_INT;
        if (pos != end && *pos == '.') {
          out.ty = L_JSON_TOKEN_FLOAT;
          ++pos;
          digits();
        }
        if (pos != end && (*pos == 'e' || *pos == 'E')) {
          out.ty = L_JSON_TOKEN_FLOAT;
          ++pos;
          if (pos != end && (*pos == '+' || *pos == '-'))
            ++pos;
          digits();
        }
        if (pos != end && *pos != ',' && *pos != ']' && *pos != '}' &&
            *pos != ' ' && *pos != '\t' && *pos != '\r' && *pos != '\n')
          throw JsonException("invalid character in json number");
        if (out.ty == L_JSON_TOKEN_INT) {
          const auto result = std::from_chars(beg, pos, out.num_int);
          if (result.ec != std::errc{} || result.ptr != pos)
            throw JsonException("json integer is outside int64 range");
        } else {
          // Classic locale keeps JSON's decimal point independent of the host.
          // Floating from_chars is not available on all supported C++17 libs.
          std::istringstream number(std::string(beg, pos));
          number.imbue(std::locale::classic());
          number >> out.num_float;
          if (number.fail() || !std::isfinite(out.num_float))
            throw JsonException("json number is outside double range");
        }
        return true;
      }

      // Try parse strings.
      if (c == '"') {
        out.ty = L_JSON_TOKEN_STRING;
        bool escape = false;
        while (++pos != end) {
          c = *pos;
          if (escape) {
            switch (c) {
              case '"':
              case '/':
              case '\\':
                break;
              case 'b':
                c = '\b';
                break;
              case 'f':
                c = '\f';
                break;
              case 'n':
                c = '\n';
                break;
              case 'r':
                c = '\r';
                break;
              case 't':
                c = '\t';
                break;
              case 'u':
                throw JsonException("unicode escape is not supported");
              default:
                throw JsonException("invalid escape charater");
            }
            escape = false;
          } else {
            if (c == '\\') {
              escape = true;
              continue;
            } else if (c == '"') {
              if (escape != false) {
                throw JsonException("incomplete escape sequence");
              }
              out.str = ss.str();
              pos += 1;
              return true;
            }
          }
          ss.put(c);
        }
        throw JsonException("unexpected end of string");
      }

      // Try parse literals.
      if (pos + 4 <= end) {
        if (unsafe_starts_with("null")) {
          out.ty = L_JSON_TOKEN_NULL;
          pos += 4;
          return true;
        }
        if (unsafe_starts_with("true")) {
          out.ty = L_JSON_TOKEN_TRUE;
          pos += 4;
          return true;
        }
      }
      if (pos + 5 <= end) {
        if (unsafe_starts_with("false")) {
          out.ty = L_JSON_TOKEN_FALSE;
          pos += 5;
          return true;
        }
      }
      throw JsonException("invalid json token");
    }
    out.ty = L_JSON_TOKEN_UNDEFINED;
    return false;
  }
};

bool try_parse_impl(Tokenizer &tokenizer,
                    JsonValue &out,
                    bool unique_keys = false) {
  JsonToken token;
  while (tokenizer.next_token(token)) {
    JsonValue val;
    switch (token.ty) {
      case L_JSON_TOKEN_TRUE:
        out.ty = L_JSON_BOOLEAN;
        out.b = true;
        return true;
      case L_JSON_TOKEN_FALSE:
        out.ty = L_JSON_BOOLEAN;
        out.b = false;
        return true;
      case L_JSON_TOKEN_NULL:
        out.ty = L_JSON_NULL;
        return true;
      case L_JSON_TOKEN_STRING:
        out.ty = L_JSON_STRING;
        out.str = std::move(token.str);
        return true;
      case L_JSON_TOKEN_INT:
        out.ty = L_JSON_INT;
        out.num_int = token.num_int;
        return true;
      case L_JSON_TOKEN_FLOAT:
        out.ty = L_JSON_FLOAT;
        out.num_float = token.num_float;
        return true;
      case L_JSON_TOKEN_OPEN_BRACKET:
        out.ty = L_JSON_ARRAY;
        for (;;) {
          if (!try_parse_impl(tokenizer, val, unique_keys)) {
            // When the array has no element.
            break;
          }
          out.arr.inner.emplace_back(std::move(val));
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_COMMA) {
              continue;
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACKET) {
              break;
            } else {
              throw JsonException("unexpected token in array");
            }
          } else {
            throw JsonException("unexpected end of array");
          }
        }
        return true;
      case L_JSON_TOKEN_OPEN_BRACE:
        out.ty = L_JSON_OBJECT;
        for (;;) {
          // Match the key.
          std::string key;
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_STRING) {
              key = std::move(token.str);
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACE) {
              // The object has no field.
              break;
            } else {
              throw JsonException("unexpected object field key type");
            }
          } else {
            throw JsonException("unexpected end of object");
          }
          // Match the colon.
          if (!tokenizer.next_token(token)) {
            throw JsonException("unexpected end of object");
          }
          if (token.ty != L_JSON_TOKEN_COLON) {
            throw JsonException("unexpected token in object");
          }
          // Match the value.
          if (!try_parse_impl(tokenizer, val, unique_keys)) {
            throw JsonException("unexpected end of object");
          }
          if (unique_keys && out.obj.inner.count(key))
            throw JsonException("duplicate object field");
          out.obj.inner[key] = std::move(val);
          // Should we head for another round?
          if (tokenizer.next_token(token)) {
            if (token.ty == L_JSON_TOKEN_COMMA) {
              continue;
            } else if (token.ty == L_JSON_TOKEN_CLOSE_BRACE) {
              break;
            } else {
              throw JsonException("unexpected token in object");
            }
          } else {
            throw JsonException("unexpected end of object");
          }
        }
        return true;
      case L_JSON_TOKEN_CLOSE_BRACE:
      case L_JSON_TOKEN_CLOSE_BRACKET:
        return false;
      default:
        throw JsonException("unexpected token");
    }
  }
  throw JsonException("unexpected program state");
}

JsonValue parse(const char *beg, const char *end) {
  if (beg == nullptr || end == nullptr || beg >= end) {
    throw JsonException("json text is empty");
  }
  JsonValue rv;
  Tokenizer tokenizer(beg, end);
  if (!try_parse_impl(tokenizer, rv)) {
    throw JsonException("unexpected close token");
  }
  return rv;
}
JsonValue parse(const std::string &json_lit) {
  return parse(json_lit.c_str(), json_lit.c_str() + json_lit.size());
}
JsonValue parse_complete(const std::string &json_lit) {
  if (json_lit.empty() || json_lit.find('\0') != std::string::npos)
    throw JsonException("json text is empty or contains NUL");
  Tokenizer tokenizer(json_lit.data(), json_lit.data() + json_lit.size());
  JsonValue result;
  if (!try_parse_impl(tokenizer, result, true))
    throw JsonException("unexpected close token");
  while (tokenizer.pos != tokenizer.end) {
    const char value = *tokenizer.pos++;
    if (value != ' ' && value != '\t' && value != '\r' && value != '\n')
      throw JsonException("unexpected content after json document");
  }
  return result;
}
bool try_parse(const std::string &json_lit, JsonValue &out) {
  try {
    out = parse(json_lit);
  } catch (const JsonException &) {
    // log::error("failed to parse json: ", e.what());
    return true;
  }
  return false;
}

void print_escaped_string(const std::string &value, std::stringstream &out) {
  out << '"';
  for (const char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  out << '"';
}

void print_impl(const JsonValue &json, std::stringstream &out) {
  switch (json.ty) {
    case L_JSON_NULL:
      out << "null";
      return;
    case L_JSON_BOOLEAN:
      out << (json.b ? "true" : "false");
      return;
    case L_JSON_FLOAT:
      out << json.num_float;
      return;
    case L_JSON_INT:
      out << json.num_int;
      return;
    case L_JSON_STRING:
      print_escaped_string(json.str, out);
      return;
    case L_JSON_OBJECT:
      out << "{";
      {
        bool is_first_iter = true;
        for (const auto &pair : json.obj.inner) {
          if (is_first_iter) {
            is_first_iter = false;
          } else {
            out << ",";
          }
          print_escaped_string(pair.first, out);
          out << ":";
          print_impl(pair.second, out);
        }
      }
      out << "}";
      return;
    case L_JSON_ARRAY:
      out << "[";
      {
        bool is_first_iter = true;
        for (const auto &elem : json.arr.inner) {
          if (is_first_iter) {
            is_first_iter = false;
          } else {
            out << ",";
          }
          print_impl(elem, out);
        }
      }
      out << "]";
      return;
  }
}
std::string print(const JsonValue &json) {
  std::stringstream ss;
  print_impl(json, ss);
  return ss.str();
}

}  // namespace json
}  // namespace liong
