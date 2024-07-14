from typing import Dict

class SymbolEncoder():
    def __init__(self):
        self.reserved_chars = 2
        self.base = 257 - self.reserved_chars
    
    def get_encoded_char(self, char_num: int) -> str:
        return chr(char_num + self.reserved_chars)

    def encode(self, decimal: int = 0):
        if decimal < self.base:
            return self.get_encoded_char(decimal)
        return self.encode(decimal // self.base) + self.get_encoded_char(decimal % self.base)

    def get_decoded_char(self, char_str) -> int:
        return ord(char_str) - self.reserved_chars
    
    def decode(self, symbol: str, pow: int = 0):
        if symbol == "":
            return 0
        return self.get_decoded_char(symbol[-1]) * (self.base ** pow) + self.decode(symbol[0:-1], pow + 1)

class RegisterQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.register = ""

    def shift(self, new_char: str) -> None:
        if len(self.register) < self.max_size:
            self.register += new_char
            return

        self.register = self.register[1:] + new_char

    def find_match(self, substring) -> bool:

        if len(self.register) < len(substring):
            return False
        
        return self.register[len(self.register)-len(substring):] == substring
    
    @property
    def get_current_register(self) -> str:
        return self.register


class DictCompress:

    def __init__(self, dictionary_path: str) -> None:

        # escape char to signal reading symbol from dictionary
        self.start_escape_char = 2
        self.end_escape_char = 1

        # load dict into memory for symbol substitution
        self.dictionary: Dict[int, str] = {}
        self.max_dict_value = 0

        symbol_encoder = SymbolEncoder()

        with open(dictionary_path, "r") as dictionary:

            # assign integer values to dictionary
            for i, dict_item in enumerate(dictionary):

                self.dictionary[symbol_encoder.encode(i)] = dict_item.replace("\n", "")

                # set max dict value
                if len(dict_item) > self.max_dict_value:
                    self.max_dict_value = len(dict_item)
                

    def compress(self, input_file_path: str, output_file_path: str) -> None:

        reg_queue = RegisterQueue(self.max_dict_value)

        matches: Dict[int, int] = {}

        with open(input_file_path, "rb") as inp:
            # read file for first iteration
            iteration_num = 0
            while True:
                in_char = inp.read(1)
                if not in_char:
                    break

                reg_queue.shift(in_char.decode())

                for dict_id, dict_item in self.dictionary.items():
                    found_match = reg_queue.find_match(dict_item)
                    if found_match:
                        matches[iteration_num+1-len(dict_item)] = dict_id
                        
                iteration_num += 1

            # jump to start of file
            inp.seek(0)
            sec_iteration_num = 0

            with open(output_file_path, "wb") as out:
                # read file for second iteration and subsitute symbols
                while True:
                    in_char = inp.read(1)
                    if not in_char:
                        break
                    if sec_iteration_num in matches:
                        symbol_bytes = (f"{chr(self.start_escape_char)}"
                                        f"{str(matches[sec_iteration_num])}"
                                        f"{chr(self.end_escape_char)}").encode()
                        out.write(symbol_bytes)
                        move_pointer = len(self.dictionary[matches[sec_iteration_num]])
                        inp.seek(move_pointer-1, 1)
                        sec_iteration_num += move_pointer
                    else:
                        out.write(in_char)
                        sec_iteration_num += 1

    def decompress(self, input_file_path: str, output_file_path: str) -> None:
        with open(input_file_path, "rb") as inp, open(output_file_path, "wb") as out:
            while True:
                in_char = inp.read(1)
                if not in_char:
                    break
                if in_char.decode() == chr(self.start_escape_char):
                    decoded_token = self.read_encoded_token(inp)
                    out.write(self.dictionary[decoded_token].encode())
                else:
                    out.write(in_char)
                    
    def read_encoded_token(self, in_file_reader) -> str:
        current_token = b""
        while True:
            in_char = in_file_reader.read(1)
            if not in_char:
                raise RuntimeError("unexpected end to file")
            decoded_in_char = in_char
            if decoded_in_char == chr(self.end_escape_char).encode():
                break
            current_token += decoded_in_char
        return current_token.decode()





